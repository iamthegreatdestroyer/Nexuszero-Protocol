<!--
Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.

This documentation is part of NexusZero Protocol.

NexusZero Protocol™, NexusZero™, and Privacy Morphing™ are trademarks of
NexusZero Protocol.

Licensed under AGPLv3. Commercial licenses available at legal@nexuszero.io.

Patent Pending.
-->

# NexusZero Protocol™ - Contributors Guide

Thank you for your interest in contributing to NexusZero Protocol! This document
outlines the contribution process, including the Contributor License Agreement (CLA)
requirements.

---

## 📋 Table of Contents

1. [Contributor License Agreement](#contributor-license-agreement)
2. [How to Contribute](#how-to-contribute)
3. [Code of Conduct](#code-of-conduct)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Recognition](#recognition)
6. [Contact](#contact)

---

## Contributor License Agreement

### Why a CLA?

NexusZero Protocol uses a dual-license model (AGPLv3 + Commercial). The CLA ensures:

1. You have the legal right to contribute the code
2. You grant necessary patent and copyright licenses
3. The project can continue to offer both license options
4. Contributors are protected from liability

### CLA Text

By submitting a pull request or other contribution to NexusZero Protocol, you agree to the following Contributor License Agreement:

---

**NEXUSZERO PROTOCOL CONTRIBUTOR LICENSE AGREEMENT**

Version 1.0 - December 2025

**1. Definitions**

"Contribution" means any original work of authorship, including any modifications
or additions to an existing work, that you submit to NexusZero Protocol.

"You" (or "Your") means the individual or legal entity making the Contribution.

"Project" means the NexusZero Protocol software and associated documentation.

**2. Grant of Copyright License**

You hereby grant to NexusZero Protocol and to recipients of software distributed
by NexusZero Protocol a perpetual, worldwide, non-exclusive, no-charge, royalty-free,
irrevocable copyright license to reproduce, prepare derivative works of, publicly
display, publicly perform, sublicense, and distribute Your Contributions and such
derivative works.

**3. Grant of Patent License**

You hereby grant to NexusZero Protocol and to recipients of software distributed
by NexusZero Protocol a perpetual, worldwide, non-exclusive, no-charge, royalty-free,
irrevocable patent license to make, have made, use, offer to sell, sell, import,
and otherwise transfer the Work, where such license applies only to those patent
claims licensable by You that are necessarily infringed by Your Contribution(s)
alone or by combination of Your Contribution(s) with the Work.

**4. Representations**

You represent that:

a) You have the legal authority to enter into this Agreement.

b) You own the copyright to Your Contribution or have permission to contribute it.

c) Your Contribution does not violate any third-party intellectual property rights.

d) If your employer has rights to intellectual property you create, you have received
permission to make Contributions, or your employer has waived such rights.

**5. Support and Warranty Disclaimer**

You are not expected to provide support for Your Contributions. You provide Your
Contributions on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

**6. Notification**

You agree to notify NexusZero Protocol of any facts or circumstances of which you
become aware that would make these representations inaccurate in any respect.

---

### How to Sign the CLA

**Option 1: Pull Request Comment**

Add the following comment to your first pull request:

```
I have read and agree to the NexusZero Protocol Contributor License Agreement
as documented in CONTRIBUTORS.md.

Signed: [Your Full Name]
Email: [Your Email]
Date: [Date]
GitHub Username: @[username]
```

**Option 2: Email**

Send a signed statement to: contributors@nexuszero.io

**Option 3: Digital Signature**

For organizations, contact legal@nexuszero.io for formal CLA execution.

---

## How to Contribute

### 1. Fork the Repository

```bash
gh repo fork nexuszero/nexuszero-protocol
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the coding standards in `.github/CODING_STANDARDS.md`
- Add tests for new functionality
- Update documentation as needed
- Add copyright headers to new files

### 4. Commit Your Changes

```bash
git commit -m "feat(scope): description of change

- Detailed bullet points
- About what changed

Signed-off-by: Your Name <your@email.com>"
```

### 5. Submit a Pull Request

- Fill out the pull request template completely
- Sign the CLA if this is your first contribution
- Ensure all CI checks pass
- Request review from maintainers

---

## Code of Conduct

### Our Pledge

We pledge to make participation in this project a harassment-free experience for
everyone, regardless of age, body size, disability, ethnicity, gender identity,
level of experience, nationality, personal appearance, race, religion, or sexual
identity and orientation.

### Our Standards

**Positive behavior:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior:**

- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations may be reported to: conduct@nexuszero.io

All complaints will be reviewed and investigated, resulting in a response deemed
necessary and appropriate. Maintainers have the right and responsibility to remove,
edit, or reject contributions not aligned with this Code of Conduct.

---

## Contribution Guidelines

### Code Quality

- **Tests Required**: All new features must include tests (90%+ coverage)
- **Documentation**: Update relevant documentation
- **Type Safety**: Use type hints (Python) or proper types (Rust/TypeScript)
- **Error Handling**: Handle errors gracefully
- **Security**: Follow secure coding practices

### Commit Messages

Follow Conventional Commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `security`

### Copyright Headers

Add copyright headers to all new source files:

```rust
// Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
//
// This file is part of NexusZero Protocol.
//
// NexusZero Protocol is dual-licensed:
// 1. AGPLv3 for open-source use
// 2. Commercial license: legal@nexuszero.io
//
// Patent Pending.
```

### Security Contributions

For security-related contributions:

1. **DO NOT** submit security vulnerabilities as public PRs
2. Email: security@nexuszero.io
3. Wait for triage before public disclosure
4. Security fixes may be expedited through the review process

---

## Recognition

### Contributors Hall of Fame

Contributors who have signed the CLA and had contributions merged:

| Contributor      | GitHub    | Contributions  | Date Joined |
| ---------------- | --------- | -------------- | ----------- |
| _Your name here_ | @username | Features/Fixes | YYYY-MM     |

### Types of Recognition

- **Code Contributors**: Listed in CONTRIBUTORS.md
- **Documentation Contributors**: Listed in CONTRIBUTORS.md
- **Security Researchers**: Listed in SECURITY.md Hall of Fame
- **Major Contributors**: Special acknowledgment in releases

---

## Contact

| Purpose                | Contact                   |
| ---------------------- | ------------------------- |
| Contribution Questions | contributors@nexuszero.io |
| CLA Inquiries          | legal@nexuszero.io        |
| Code of Conduct        | conduct@nexuszero.io      |
| General                | hello@nexuszero.io        |

---

## Document Information

| Field        | Value            |
| ------------ | ---------------- |
| Document     | CONTRIBUTORS.md  |
| Version      | 1.0.0            |
| Created      | December 2, 2025 |
| Last Updated | December 2, 2025 |
| Status       | Active           |

---

**© 2025 NexusZero Protocol. All Rights Reserved.**

**NexusZero Protocol™ is a trademark of NexusZero Protocol.**
