#!/usr/bin/env node
/**
 * Generic test user setup script for EvalView
 * Works with any database/API setup
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise((resolve) => rl.question(prompt, resolve));
}

async function main() {
  console.log('┌─────────────────────────────────────────────┐');
  console.log('│   EvalView - Test User Setup               │');
  console.log('└─────────────────────────────────────────────┘\n');

  console.log('This script helps you configure a test user for EvalView.\n');

  const setupType = await question(
    'How would you like to set up the test user?\n' +
    '  1. Use a fixed test user ID (e.g., "test-user")\n' +
    '  2. Use an existing user ID from your system\n' +
    '  3. Skip - I\'ll handle this manually\n' +
    'Choice (1-3): '
  );

  let userId = 'test-user';
  let configPath = path.join(process.cwd(), '.evalview', 'config.yaml');

  if (setupType.trim() === '2') {
    userId = await question('Enter your existing user ID: ');
    userId = userId.trim();
  } else if (setupType.trim() === '3') {
    console.log('\n✅ Skipping test user setup.');
    console.log('Remember to configure userId in your test cases!\n');
    rl.close();
    return;
  }

  // Update test cases with the user ID
  const testCasesDir = path.join(process.cwd(), 'tests', 'test-cases');

  if (fs.existsSync(testCasesDir)) {
    const updateTests = await question(
      `\nUpdate all test cases in tests/test-cases/ to use userId: "${userId}"? (y/n): `
    );

    if (updateTests.toLowerCase() === 'y') {
      const files = fs.readdirSync(testCasesDir).filter(f => f.endsWith('.yaml'));

      for (const file of files) {
        const filePath = path.join(testCasesDir, file);
        let content = fs.readFileSync(filePath, 'utf8');

        // Add userId to context if not present, or update if present
        if (content.includes('context:')) {
          if (content.includes('userId:')) {
            // Update existing userId
            content = content.replace(/userId:\s*"[^"]*"/, `userId: "${userId}"`);
            content = content.replace(/userId:\s*'[^']*'/, `userId: "${userId}"`);
          } else {
            // Add userId to existing context
            content = content.replace(
              /context:\s*\n/,
              `context:\n    userId: "${userId}"\n`
            );
          }
        } else {
          // Add context with userId
          content = content.replace(
            /input:\s*\n(\s+)query:/,
            `input:\n$1query:`
          );
          content = content.replace(
            /(\s+)query:([^\n]*)\n/,
            `$1query:$2\n$1context:\n$1  userId: "${userId}"\n`
          );
        }

        fs.writeFileSync(filePath, content);
        console.log(`  ✅ Updated ${file}`);
      }
    }
  }

  // Create a .evalview/test-user.txt file for reference
  const testUserFile = path.join(process.cwd(), '.evalview', 'test-user.txt');
  fs.writeFileSync(testUserFile, userId);

  console.log(`\n✅ Test user configured: ${userId}`);
  console.log(`   Saved to: .evalview/test-user.txt\n`);

  console.log('Next steps:');
  console.log('  1. Review your test cases in tests/test-cases/');
  console.log('  2. Run: evalview run --verbose');
  console.log('  3. See DEBUGGING.md if you encounter issues\n');

  rl.close();
}

main().catch(error => {
  console.error('Error:', error.message);
  process.exit(1);
});
