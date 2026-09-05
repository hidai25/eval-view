// CommonJS for actions/github-script even when the repository uses type=module.
// A green core run is not live-provider recovery.
const fs = require('fs');

const SCOPES = {
  core: {
    title: '🐕 Core dogfood is failing',
    label: 'dogfood-core',
    checks: ['pytest', 'mypy', 'demo', 'help', 'dogfood-e2e', 'regression', 'monitor'],
    coverage: 'deterministic core checks',
    disclaimer: 'Live provider coverage is reported separately; this run does not establish live-provider recovery.',
  },
  live: {
    title: '🐕 Live provider dogfood is failing',
    label: 'dogfood-live',
    checks: ['provider', 'pytest-llm', 'dogfood'],
    coverage: 'live provider checks',
    disclaimer: 'This issue tracks live provider coverage only; core correctness has a separate workflow.',
  },
};

module.exports = async function updateIssue({github, context, core}) {
  // Defense in depth: issue writes are never performed for PR or branch runs.
  if (context.eventName === 'pull_request' || context.eventName === 'pull_request_target'
      || context.payload?.pull_request || context.ref !== 'refs/heads/main') {
    core.info('Skipping issue updates outside non-PR main runs.');
    return;
  }
  const scope = process.env.DOGFOOD_SCOPE;
  const config = SCOPES[scope];
  if (!config) throw new Error('DOGFOOD_SCOPE must be explicitly set to core or live');

  // Unknown or missing summary evidence fails closed. Reject unrelated log names
  // so core issue comments can never accidentally publish live provider results.
  const reported = (process.env.FAILED_CHECKS || '').split(',').filter(Boolean);
  const invalid = reported.some(check => !config.checks.includes(check));
  const hasFailures = process.env.HAS_FAILURES !== 'false' || reported.length > 0;
  const failed = invalid || (hasFailures && reported.length === 0)
    ? ['workflow-setup'] : reported;
  const date = new Date().toISOString().split('T')[0];
  const runUrl = `${process.env.GITHUB_SERVER_URL}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;

  for (const label of ['dogfood', config.label]) {
    try {
      await github.rest.issues.createLabel({
        ...context.repo, name: label, color: 'FFA500', description: 'Dogfood check failures',
      });
    } catch (error) {
      if (error.status !== 422) throw error;
    }
  }
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...context.repo, state: 'open', labels: config.label, per_page: 100,
  });
  const rolling = issues.find(issue => !issue.pull_request && issue.title === config.title
    && issue.labels.some(label => (typeof label === 'string' ? label : label.name) === config.label));

  if (!hasFailures) {
    if (rolling) {
      await github.rest.issues.createComment({
        ...context.repo, issue_number: rolling.number,
        body: `✅ All ${config.coverage} passed on ${date}.\n\n${config.disclaimer}\n\n[Run](${runUrl})`,
      });
      await github.rest.issues.update({
        ...context.repo, issue_number: rolling.number, state: 'closed',
      });
    }
    return;
  }

  let body = `## ${date}\n\nFailed or blocked ${scope} checks: ${failed.map(check => `\`${check}\``).join(', ')}\n\n${config.disclaimer}\n\n`;
  if (scope === 'live' && failed.includes('provider')) {
    body += 'Provider preflight failed. Live results are unavailable, not passing. Restore credentials, quota, or availability and rerun.\n\n';
  }
  for (const check of failed) {
    body += `### ❌ ${check}\n\n`;
    try {
      const output = fs.readFileSync(`${check}-output.txt`, 'utf8');
      body += '```text\n' + output.slice(-3000) + '\n```\n\n';
    } catch (error) {
      body += 'No output recorded; this check may be blocked or not started. See the workflow evidence.\n\n';
    }
  }
  body += `Full logs and reports: **${scope}-dogfood-evidence-${context.runId}** artifact.\n\n[View workflow run](${runUrl})`;
  if (rolling) {
    await github.rest.issues.createComment({...context.repo, issue_number: rolling.number, body});
  } else {
    await github.rest.issues.create({
      ...context.repo, title: config.title, labels: ['dogfood', config.label], body,
    });
  }
};
