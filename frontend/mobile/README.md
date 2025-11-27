# NexusZero Mobile App

Privacy-preserving transactions on iOS and Android, powered by quantum-resistant zero-knowledge proofs.

## Features

- 🔐 **Multi-wallet Management** - Create, import, and manage multiple wallets
- 🛡️ **6-Level Privacy System** - From transparent to sovereign privacy
- 💸 **Private Transactions** - Send, receive, shield, and unshield assets
- 🌉 **Cross-chain Bridge** - Bridge assets across supported chains
- 🔑 **Secure Storage** - Biometric authentication and encrypted key storage
- 📊 **Real-time Dashboard** - View balances, transactions, and privacy status

## Tech Stack

- **Framework:** React Native with Expo
- **Navigation:** React Navigation 6
- **State Management:** Zustand
- **Data Fetching:** TanStack Query (React Query)
- **Styling:** React Native StyleSheet
- **Crypto:** ethers.js

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Expo CLI
- iOS Simulator (Mac) or Android Emulator

### Installation

```bash
# Navigate to mobile directory
cd frontend/mobile

# Install dependencies
npm install

# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android
```

### Development

```bash
# Start Expo development server
npx expo start

# Press 'i' for iOS simulator
# Press 'a' for Android emulator
# Scan QR code with Expo Go app on device
```

## Project Structure

```
frontend/mobile/
├── app.json              # Expo configuration
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
├── index.ts              # App entry point
├── assets/               # Images, fonts, icons
└── src/
    ├── App.tsx           # Main app component
    ├── screens/          # Screen components
    │   ├── HomeScreen.tsx
    │   ├── WalletScreen.tsx
    │   ├── TransactScreen.tsx
    │   └── SettingsScreen.tsx
    ├── components/       # Reusable components
    ├── hooks/            # Custom hooks
    ├── lib/              # Utilities
    ├── services/         # API services
    ├── store/            # Zustand stores
    └── types/            # TypeScript types
```

## Privacy Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Transparent | Public blockchain parity |
| 1 | Pseudonymous | Address obfuscation |
| 2 | Confidential | Encrypted amounts |
| 3 | Private | Full transaction privacy |
| 4 | Anonymous | Unlinkable transactions |
| 5 | Sovereign | Maximum privacy, ZK everything |

## Building for Production

```bash
# Build for iOS
npx eas build --platform ios

# Build for Android
npx eas build --platform android
```

## Security

- All private keys are stored in device secure enclave
- Biometric authentication required for transactions
- Auto-lock when app is backgrounded
- No private data sent to external servers

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT OR Apache-2.0
