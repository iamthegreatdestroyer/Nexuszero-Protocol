<!--
Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.

This documentation is part of NexusZero Protocol.

NexusZero Protocol™, NexusZero™, and Privacy Morphing™ are trademarks of
NexusZero Protocol.

Licensed under AGPLv3. Commercial licenses available at licensing@nexuszero.io.

Patent Pending.
-->

# Contributing to NexusZero Protocol

Thank you for considering contributing to NexusZero Protocol! This document provides
guidelines for contributing to the project.

---

## 📋 Quick Links

- [Contributor License Agreement](CONTRIBUTORS.md#contributor-license-agreement)
- [Code of Conduct](CONTRIBUTORS.md#code-of-conduct)
- [Security Policy](SECURITY.md)
- [License Information](LICENSE)

---

## ⚠️ Important: Intellectual Property Agreement

### Contributor License Agreement (CLA)

**All contributions require acceptance of our Contributor License Agreement (CLA).**

By submitting a pull request, you agree to the following:

1. **Copyright Assignment**: You grant NexusZero Protocol a perpetual, worldwide,
   non-exclusive, royalty-free license to use, modify, and distribute your contribution
   under both the AGPLv3 and commercial licenses.

2. **Patent Grant**: You grant a patent license for any patents covering your
   contribution, protecting both the project and other contributors.

3. **Original Work**: You certify that your contribution is your original work,
   or you have the right to submit it under the project's license.

4. **No Warranty**: You understand that contributions are provided "as-is" without
   warranty.

The full CLA text is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## 🚀 Getting Started

### Prerequisites

- **Rust** 1.70+ (for core cryptography)
- **Python** 3.10+ (for neural optimizer)
- **Node.js** 18+ (for SDK and tooling)
- **Docker** (for services)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# Initialize the development environment
./scripts/init-project.ps1

# Run tests
cargo test --all
pytest nexuszero-optimizer/
```

---

## 🔀 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Follow the coding standards for each language (see below)
- Add tests for new functionality
- Update documentation as needed
- **Add copyright headers to new files** (see templates in `legal/templates/`)

### 3. Commit with Signed-off-by

```bash
git commit -s -m "feat: add new feature description"
```

The `-s` flag adds a `Signed-off-by` line, indicating CLA acceptance.

### 4. Submit Pull Request

- Fill out the PR template completely
- Reference any related issues
- Ensure CI passes

---

## 📝 Coding Standards

### Rust

- Use `rustfmt` for formatting
- Use `clippy` for linting
- Follow Rust API guidelines
- Add `//!` doc comments to modules
- Add `///` doc comments to public items

### TypeScript/JavaScript

- Use ESLint and Prettier
- Prefer TypeScript over JavaScript
- Use strict type checking

### Python

- Use Black for formatting
- Use MyPy for type checking
- Follow PEP 8 guidelines

### Solidity

- Follow Solidity style guide
- Use Slither and Foundry for testing
- Add NatSpec comments

---

## 📄 Copyright Headers

All source files must include a copyright header. Templates are in `legal/templates/copyright_header.txt`.

### Rust Example

```rust
// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.
```

### TypeScript/JavaScript Example

```typescript
/**
 * Copyright (c) 2025 NexusZero Protocol
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
 * Licensed under the GNU Affero General Public License v3.0 or later.
 * Commercial licensing available at https://nexuszero.io/licensing
 *
 * NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
 * are trademarks of NexusZero Protocol. All Rights Reserved.
 */
```

### Python Example

```python
# Copyright (c) 2025 NexusZero Protocol
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
# Licensed under the GNU Affero General Public License v3.0 or later.
# Commercial licensing available at https://nexuszero.io/licensing
#
# NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
# are trademarks of NexusZero Protocol. All Rights Reserved.
```

---

## 🏷️ Trademark Usage

When contributing documentation or comments that reference NexusZero Protocol,
please follow these guidelines:

- Use **NexusZero Protocol™** for the first reference in a document
- Subsequent references can use **NexusZero** or **the Protocol**
- Never use the trademarks in a way that suggests endorsement
- See [legal/trademarks/TRADEMARK_GUIDELINES.md](legal/trademarks/TRADEMARK_GUIDELINES.md)

---

## 🔒 Security

If you discover a security vulnerability, please follow our [Security Policy](SECURITY.md).

**DO NOT** open a public issue for security vulnerabilities.

---

## 📬 Contact

- **General Questions**: Create a GitHub Discussion
- **Security Issues**: security@nexuszero.io
- **Licensing Questions**: licensing@nexuszero.io
- **Legal Matters**: legal@nexuszero.io

---

## 🙏 Recognition

Contributors who make significant contributions will be recognized in:

- The [CONTRIBUTORS.md](CONTRIBUTORS.md#core-contributors) file
- Release notes
- Project documentation

---

**Thank you for contributing to NexusZero Protocol!**

_By contributing, you agree to the terms outlined in this document and the CLA._
