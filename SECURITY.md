# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of EvalView seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Recommended): Use the [Security Advisory](https://github.com/hidai25/EvalView/security/advisories/new) feature
2. **Email**: Send an email to the project maintainers (you can find contact information in the project repository)

### What to Include

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 5 business days with an assessment of the report
- **Resolution**: We aim to release patches for verified vulnerabilities as quickly as possible, typically within 30 days

### What to Expect

After you submit a report, we will:

1. Confirm receipt of your vulnerability report
2. Assess the impact and severity of the vulnerability
3. Work on a fix and coordinate a release timeline
4. Keep you informed of our progress
5. Credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Users

When using EvalView, please follow these security best practices:

### API Keys and Secrets

- **Never commit API keys**: Always use environment variables or `.env` files (which are gitignored)
- **Rotate keys regularly**: Rotate OpenAI API keys and other credentials periodically
- **Use least privilege**: Grant API keys only the minimum required permissions

### Test Case Security

- **Sanitize test data**: Avoid including sensitive data in test cases
- **Review before sharing**: Ensure test cases don't contain proprietary information
- **Validate inputs**: When writing custom adapters, validate and sanitize all inputs

### Agent Security

- **Isolate test environments**: Run agent tests in isolated/sandboxed environments
- **Monitor costs**: Set up billing alerts for API providers
- **Review agent actions**: Regularly audit tool calls and agent behaviors in traces

### Dependencies

- **Keep updated**: Regularly update EvalView and its dependencies
- **Review dependencies**: Use tools like `pip-audit` to check for known vulnerabilities
- **Lock versions**: Use `requirements.txt` or `poetry.lock` to pin dependency versions

## Known Security Considerations

### LLM-as-Judge Evaluation

- EvalView uses OpenAI's API for output quality evaluation
- Test outputs and expected outputs are sent to OpenAI for comparison
- **Recommendation**: Don't include sensitive/proprietary data in test cases if using LLM-as-judge

### HTTP Adapters

- Custom HTTP adapters may expose your agent endpoints
- **Recommendation**: Use authentication, HTTPS, and rate limiting on agent endpoints

### Trace Data

- Execution traces may contain sensitive information from agent responses
- **Recommendation**: Sanitize traces before sharing or storing long-term

## Security Updates

We will disclose security vulnerabilities through:

1. **GitHub Security Advisories**: Primary notification channel
2. **Release Notes**: Documented in CHANGELOG.md
3. **GitHub Releases**: Tagged releases with security patch notes

## Attribution

We appreciate the efforts of security researchers who help keep EvalView and its users safe. With your permission, we will acknowledge your contribution in:

- Security advisories
- Release notes
- A dedicated SECURITY_CREDITS.md file (if applicable)

## Policy Updates

This security policy may be updated from time to time. Please check back periodically for any changes.

---

**Last Updated**: 2025-01-24
