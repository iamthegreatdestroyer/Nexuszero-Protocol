# Installation

## Prerequisites

Nexuszero SDK requires:
- **Node.js** 16.x or higher
- **npm** 7.x or higher (or yarn/pnpm)
- **TypeScript** 5.x (optional, for TypeScript projects)

Check your versions:

```bash
node --version   # Should be v16.x or higher
npm --version    # Should be 7.x or higher
```

## Using npm

```bash
npm install nexuszero-sdk
```

## Using yarn

```bash
yarn add nexuszero-sdk
```

## Using pnpm

```bash
pnpm add nexuszero-sdk
```

## TypeScript Configuration

If you're using TypeScript, ensure your `tsconfig.json` includes:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020"],
    "moduleResolution": "node",
    "esModuleInterop": true
  }
}
```

## Verify Installation

Create a test file to verify the installation:

```typescript
// test.ts or test.js
import { NexuszeroClient } from 'nexuszero-sdk';

const client = new NexuszeroClient();
console.log('Nexuszero SDK installed successfully!');
console.log('Parameters:', client.getParameters());
```

Run it:

```bash
node test.js           # JavaScript
npx tsx test.ts        # TypeScript (requires tsx)
```

You should see output confirming the SDK is working.

## Development Setup

For local development with the SDK source:

```bash
# Clone the repository
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol/nexuszero-sdk

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test
```

## Browser Support

::: warning
Browser support via WASM is currently in development. The SDK currently supports Node.js environments only.
:::

Stay tuned for browser support updates.

## Troubleshooting

### Module not found errors

If you encounter module resolution errors:

```bash
# Clear npm cache
npm cache clean --force

# Reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript errors

Ensure you have the latest version:

```bash
npm install -D typescript@latest
```

### BigInt support

The SDK uses BigInt, which requires Node.js 10.4.0+. If you see BigInt errors:

```bash
# Update Node.js
nvm install node
nvm use node
```

## Next Steps

- [Getting Started Guide](/guide/getting-started)
- [API Reference](/api/client)
- [Examples](/examples/age-verification)
