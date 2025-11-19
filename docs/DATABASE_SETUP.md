# Database Setup for AgentEval

AgentEval can test any agent that uses a database for user management. This guide shows you how to set up test users for different database systems.

## Overview

Many AI agents require a valid user ID when processing requests. AgentEval needs a test user in your database to run tests without affecting production users.

## Quick Setup

### Option 1: Use the Setup Script (Recommended)

```bash
cd /path/to/EvalView
node scripts/setup-test-user.js
```

This interactive script will:
1. Ask if you want to use a fixed test user or existing user
2. Update all your test cases with the chosen user ID
3. Save the configuration

### Option 2: Manual Configuration

Edit your test cases to include a `userId`:

```yaml
# tests/test-cases/example.yaml
input:
  query: "Your test query"
  context:
    userId: "your-test-user-id"  # Add this line
```

## Database-Specific Guides

### PostgreSQL (Prisma)

If your agent uses Prisma with PostgreSQL:

1. **Create a seed script** in your agent project:

```typescript
// prisma/seed-test-user.ts
import { config } from 'dotenv';
import { PrismaClient } from '@prisma/client';

config({ path: '.env.local' });
const prisma = new PrismaClient();

async function main() {
  const testUser = await prisma.user.upsert({
    where: { email: 'test@agenteval.local' },
    update: { name: 'AgentEval Test User' },
    create: {
      id: 'test-user',
      email: 'test@agenteval.local',
      name: 'AgentEval Test User',
      // Add other required fields based on your schema
    },
  });
  console.log('Test user created:', testUser.id);
}

main().then(() => prisma.$disconnect());
```

2. **Run the seed script:**

```bash
npx tsx prisma/seed-test-user.ts
```

3. **Configure AgentEval:**

```bash
cd /path/to/EvalView
node scripts/setup-test-user.js
# Choose option 1 and enter: test-user
```

### MongoDB

For MongoDB-based agents:

```javascript
// scripts/create-test-user.js
const { MongoClient } = require('mongodb');
require('dotenv').config({ path: '.env.local' });

async function main() {
  const client = new MongoClient(process.env.MONGODB_URI);
  await client.connect();

  const db = client.db();
  const users = db.collection('users');

  await users.updateOne(
    { email: 'test@agenteval.local' },
    {
      $set: {
        _id: 'test-user',
        email: 'test@agenteval.local',
        name: 'AgentEval Test User',
        createdAt: new Date()
      }
    },
    { upsert: true }
  );

  console.log('Test user created: test-user');
  await client.close();
}

main();
```

### MySQL

For MySQL databases:

```sql
-- create_test_user.sql
INSERT INTO users (id, email, name, created_at)
VALUES ('test-user', 'test@agenteval.local', 'AgentEval Test User', NOW())
ON DUPLICATE KEY UPDATE
  name = 'AgentEval Test User';
```

Run with:
```bash
mysql -u username -p database_name < create_test_user.sql
```

### Firebase / Firestore

For Firebase Auth:

```javascript
// scripts/create-test-user.js
const admin = require('firebase-admin');
const serviceAccount = require('./serviceAccountKey.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

async function createTestUser() {
  try {
    const user = await admin.auth().createUser({
      uid: 'test-user',
      email: 'test@agenteval.local',
      displayName: 'AgentEval Test User'
    });
    console.log('Test user created:', user.uid);
  } catch (error) {
    if (error.code === 'auth/uid-already-exists') {
      console.log('Test user already exists: test-user');
    } else {
      throw error;
    }
  }
}

createTestUser();
```

### Supabase

For Supabase:

```sql
-- Run in Supabase SQL Editor
INSERT INTO auth.users (
  id,
  instance_id,
  email,
  encrypted_password,
  email_confirmed_at,
  created_at,
  updated_at
)
VALUES (
  'test-user'::uuid,
  '00000000-0000-0000-0000-000000000000'::uuid,
  'test@agenteval.local',
  crypt('test-password', gen_salt('bf')),
  now(),
  now(),
  now()
)
ON CONFLICT (id) DO NOTHING;
```

## Using Existing Users

If you prefer to use an existing user instead of creating a test user:

### 1. Find Your User ID

**PostgreSQL/Prisma:**
```bash
npx prisma studio
# Or via CLI:
psql -d your_database -c "SELECT id, email FROM users WHERE email = 'your@email.com';"
```

**MongoDB:**
```javascript
db.users.findOne({ email: "your@email.com" }, { _id: 1, email: 1 })
```

**MySQL:**
```sql
SELECT id, email FROM users WHERE email = 'your@email.com';
```

### 2. Configure AgentEval

```bash
node scripts/setup-test-user.js
# Choose option 2 and enter your user ID
```

## Best Practices

### For Development

- Use a dedicated test user (`test-user`)
- Keep test user data separate from production
- Use a test-specific email domain (e.g., `@agenteval.local`)

### For CI/CD

- Automate test user creation in your CI pipeline
- Use environment-specific user IDs
- Clean up test data after runs (optional)

### For Production Testing

- Never use production user accounts
- Create a dedicated test environment
- Use separate databases for testing

## Troubleshooting

### "Foreign key constraint violated"

**Cause:** Test user doesn't exist in database

**Fix:** Run the appropriate seed script for your database type

### "User not found" Errors

**Cause:** User ID doesn't match what's in the database

**Fix:**
1. Verify user exists: check your database
2. Update test cases with correct user ID:
   ```bash
   node scripts/setup-test-user.js
   ```

### Permission Errors

**Cause:** Test user lacks necessary permissions

**Fix:** Grant appropriate roles/permissions:

```sql
-- PostgreSQL example
UPDATE users SET role = 'user' WHERE id = 'test-user';
```

## No Database? No Problem!

If your agent doesn't use a database for users, you can:

1. **Skip user ID entirely** - omit `userId` from test cases
2. **Use any string** - some APIs accept any user identifier
3. **Configure in adapter** - modify the adapter to handle userless requests

See [ADAPTERS.md](ADAPTERS.md) for custom adapter development.

## Example Test Case

After setup, your test cases should look like:

```yaml
name: "My Test"
input:
  query: "Test query"
  context:
    userId: "test-user"  # Or your chosen user ID

expected:
  tools: [...]
  output:
    contains: [...]

thresholds:
  min_score: 70
```

## Next Steps

- **Run tests:** `agent-eval run --verbose`
- **See results:** Check `.agenteval/results/`
- **Debug issues:** See [DEBUGGING.md](../DEBUGGING.md)

## Need Help?

- Check [DEBUGGING.md](../DEBUGGING.md) for common issues
- See examples in `tests/test-cases/`
- Open an issue on GitHub
