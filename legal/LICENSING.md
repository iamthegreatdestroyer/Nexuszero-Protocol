# Licensing Guide

**Project**: NexusZero Protocol  
**Primary License**: MIT License  
**Last Updated**: November 23, 2025

---

## Overview

This document describes the licensing strategy for NexusZero Protocol, including:
- Core project license (MIT)
- Commercial licensing options
- Contributor license agreements
- Third-party licenses
- Dual licensing strategy (future)

---

## Core Project License: MIT

### Why MIT?

NexusZero Protocol is licensed under the **MIT License**, one of the most permissive open source licenses.

**Advantages**:
- ✓ Encourages widespread adoption and contribution
- ✓ Allows commercial use without restrictions
- ✓ Simple and easy to understand
- ✓ Compatible with most other open source licenses
- ✓ Industry standard for cryptography and blockchain projects
- ✓ No copyleft requirements (can be used in proprietary software)

**Disadvantages**:
- Allows competitors to use and improve upon our work
- No requirement to contribute improvements back
- No patent protection clauses (unlike Apache 2.0)

**Why We Chose MIT Despite Disadvantages**:
1. **Adoption First**: Maximum adoption is more valuable than restriction
2. **Community Building**: Encourages a vibrant developer community
3. **Network Effects**: More users = more value for everyone
4. **Patent Strategy**: We protect innovations with patents, not licenses
5. **Commercial Licensing**: Premium features can be dual-licensed

### MIT License Text

See [LICENSE](../LICENSE) file for the full MIT License text.

```
MIT License

Copyright (c) 2025 Steve (iamthegreatdestroyer)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text in LICENSE file]
```

---

## Commercial Licensing (Future)

### Dual Licensing Strategy

While the core protocol is MIT licensed, we plan to offer **commercial licenses** for:

1. **Premium Features**:
   - Advanced neural optimizer models
   - Enterprise-grade SLA support
   - Priority feature development
   - Custom integration assistance

2. **Proprietary Modules**:
   - Regulatory compliance dashboard
   - Enterprise audit tools
   - Advanced monitoring and analytics
   - Custom proof templates

3. **Support and Services**:
   - 24/7 premium support
   - Training and consulting
   - Custom development
   - Security audits and certifications

### Commercial License Terms (Proposed)

**Enterprise License**:
- **Price**: $500K-$2M annually (based on scale)
- **Includes**: Premium features, priority support, custom development credits
- **Term**: 1-3 years, auto-renewing
- **Users**: Unlimited within organization
- **Modifications**: Allowed, must remain proprietary

**OEM License**:
- **Price**: Negotiable (typically $1M-$10M)
- **Includes**: Right to embed in proprietary products
- **Term**: Perpetual with maintenance
- **Support**: Full source code access and support
- **Modifications**: Allowed, proprietary or open source

**Government License**:
- **Price**: $5M-$50M (multi-year contracts)
- **Includes**: Custom deployments, security clearances, compliance
- **Term**: 5-10 years
- **Support**: Dedicated team, on-site if needed
- **Modifications**: Allowed, classified or unclassified

### Contact for Commercial Licensing

- **Email**: enterprise@nexuszero.io
- **Sales**: sales@nexuszero.io
- **Partnerships**: partnerships@nexuszero.io

---

## Contributor License Agreement (CLA)

### Why We Require a CLA

A Contributor License Agreement ensures:
1. **Legal Clarity**: Clear ownership of contributions
2. **Patent Protection**: Contributors grant patent licenses
3. **Commercial Use**: Enables commercial licensing
4. **IP Protection**: Protects against future IP disputes
5. **Continuity**: Ensures project can continue regardless of contributor status

### Types of CLAs

**Individual CLA**:
- For individual contributors
- Signs on behalf of themselves
- Required before first contribution accepted

**Corporate CLA**:
- For employees of corporations
- Corporation signs on behalf of employees
- Required if contributing as part of employment

### CLA Terms (Summary)

By signing the CLA, contributors:

1. **Grant Copyright License**: License contributions under MIT License
2. **Grant Patent License**: License any patents covering contributions
3. **Represent Original Work**: Confirm they have rights to contribute
4. **Allow Commercial Use**: Permit commercial use and dual licensing

### How to Sign the CLA

1. **Automated**: CLA Assistant bot comments on first PR
2. **Click Link**: Follow link to sign electronically
3. **Takes 2 Minutes**: Simple form, electronic signature
4. **One Time**: Only need to sign once for all contributions

### CLA Templates

- **Individual CLA**: [legal/templates/INDIVIDUAL_CLA.md](templates/) (to be created)
- **Corporate CLA**: [legal/templates/CORPORATE_CLA.md](templates/) (to be created)

---

## Third-Party Licenses

### Dependencies and Their Licenses

NexusZero Protocol depends on various open source libraries. We ensure all dependencies are MIT-compatible.

#### Rust Dependencies

| Dependency | License | Use Case | Compatible? |
|------------|---------|----------|-------------|
| ndarray | MIT/Apache-2.0 | Numerical computing | ✓ Yes |
| rand | MIT/Apache-2.0 | Random number generation | ✓ Yes |
| num-bigint | MIT/Apache-2.0 | Big integer arithmetic | ✓ Yes |
| sha3 | MIT/Apache-2.0 | SHA-3 hashing | ✓ Yes |
| blake3 | CC0-1.0/Apache-2.0 | Blake3 hashing | ✓ Yes |
| zeroize | MIT/Apache-2.0 | Secure memory clearing | ✓ Yes |
| serde | MIT/Apache-2.0 | Serialization | ✓ Yes |
| tokio | MIT | Async runtime | ✓ Yes |

#### Python Dependencies

| Dependency | License | Use Case | Compatible? |
|------------|---------|----------|-------------|
| PyTorch | BSD-3-Clause | Deep learning | ✓ Yes |
| NumPy | BSD-3-Clause | Numerical computing | ✓ Yes |
| SciPy | BSD-3-Clause | Scientific computing | ✓ Yes |
| pandas | BSD-3-Clause | Data manipulation | ✓ Yes |
| scikit-learn | BSD-3-Clause | Machine learning | ✓ Yes |

#### TypeScript/JavaScript Dependencies

| Dependency | License | Use Case | Compatible? |
|------------|---------|----------|-------------|
| React | MIT | UI framework | ✓ Yes |
| TypeScript | Apache-2.0 | Type safety | ✓ Yes |
| ethers.js | MIT | Web3 integration | ✓ Yes |
| Web3.js | LGPL-3.0 | Web3 integration | ⚠️ Review needed |

### License Compatibility Matrix

| Dependency License | MIT Compatible? | Notes |
|--------------------|-----------------|-------|
| MIT | ✓ | Perfect match |
| Apache-2.0 | ✓ | Compatible, adds patent protection |
| BSD-2/3-Clause | ✓ | Compatible, similar to MIT |
| ISC | ✓ | Compatible, equivalent to MIT |
| CC0-1.0 | ✓ | Public domain dedication |
| LGPL | ⚠️ | Dynamic linking OK, static linking requires review |
| MPL-2.0 | ⚠️ | File-level copyleft, requires review |
| GPL | ✗ | Copyleft, incompatible with MIT for derived works |
| AGPL | ✗ | Strong copyleft, incompatible |
| Proprietary | ✗ | Requires commercial license |

### Automated License Checking

**Rust**:
```bash
# Check all dependency licenses
cargo install cargo-license
cargo license

# Generate third-party license report
cargo install cargo-about
cargo about generate about.hbs > THIRD_PARTY_LICENSES.html
```

**Python**:
```bash
# Check all dependency licenses
pip install pip-licenses
pip-licenses --format=markdown --output-file=THIRD_PARTY_LICENSES_PYTHON.md
```

**Node.js**:
```bash
# Check all dependency licenses
npm install --save-dev license-checker
npx license-checker --summary > THIRD_PARTY_LICENSES_NODE.txt
```

### Adding New Dependencies

**Process**:
1. **Check License**: Verify license is MIT-compatible
2. **Legal Review**: If unsure, consult legal team (legal@nexuszero.io)
3. **Document**: Add to this document and THIRD_PARTY_LICENSES
4. **Attribute**: Add proper attribution in NOTICE file if required

**Red Flags**:
- GPL or AGPL licenses (strong copyleft)
- Proprietary licenses
- "Non-commercial use only" clauses
- Restrictions on modifications
- Export control restrictions

---

## Patent Licenses

### Outbound Patent License (MIT)

The MIT License does NOT include an explicit patent license, but legal precedent suggests it includes an implied patent license for patents covering the licensed software.

### Patent Protection Strategy

Since MIT doesn't have explicit patent clauses, we protect our innovations through:

1. **Patent Filings**: File patents on core innovations (see [INNOVATION_LOG.md](INNOVATION_LOG.md))
2. **Patent Pools**: Join defensive patent pools (OIN, COPA)
3. **Patent Licensing**: Offer patent licenses as part of commercial agreements
4. **Defensive Termination**: Reserve right to terminate license for patent aggressors

### Patent Non-Assertion Pledge (Proposed)

We commit to NOT asserting our patents against:
1. Open source implementations of our protocols
2. Non-commercial use
3. Educational and research use
4. Defensive purposes (if we're sued first)

**Exceptions**:
- Commercial competitors who sue us first
- Patent trolls and non-practicing entities
- Parties who refuse reasonable licensing terms

---

## Trademark License

### Trademarks

The following are trademarks of NexusZero Protocol:
- NexusZero™
- NexusZero Protocol™
- [Logo designs]

### Trademark Usage Guidelines

**Allowed Uses** (no permission needed):
- Referring to the NexusZero Protocol in text
- Linking to nexuszero.io
- Using in documentation or tutorials
- Indicating compatibility ("Works with NexusZero")

**Requires Permission**:
- Using trademarks in product names
- Using trademarks in domain names
- Using logos in commercial products
- Implying endorsement or affiliation

**Prohibited Uses**:
- Modifying our trademarks
- Using in a misleading way
- Using in a way that disparages the project
- Registering similar trademarks

### How to Request Trademark Permission

Email: trademarks@nexuszero.io

Include:
- Intended use of trademark
- Context (product, website, documentation)
- Duration of use
- Commercial or non-commercial

Response typically within 5 business days.

---

## License Compliance

### For Users of NexusZero Protocol

**Requirements under MIT License**:
1. **Include License**: Include copy of MIT License in distributions
2. **Include Copyright**: Include copyright notice in distributions
3. **No Warranty Disclaimer**: Retain "AS IS" disclaimer

**Optional (but appreciated)**:
- Attribute NexusZero Protocol in documentation
- Link back to our repository
- Contribute improvements back to the project

### For Contributors

**Requirements**:
1. **Sign CLA**: Sign Contributor License Agreement
2. **Original Work**: Only contribute your original work
3. **Respect Licenses**: Don't copy code from incompatible licenses
4. **Attribution**: Properly attribute any third-party code

### For Integrators

**Best Practices**:
1. **Check License Compatibility**: Ensure compatibility with your license
2. **Attribution**: Give proper credit to NexusZero Protocol
3. **Security**: Report security vulnerabilities responsibly
4. **Contributions**: Consider contributing improvements back

---

## License Violations

### Reporting License Violations

If you believe someone is violating the MIT License:

1. **Email**: legal@nexuszero.io
2. **Include**:
   - Description of violation
   - Evidence (links, screenshots)
   - Contact information of violator (if known)
   - Your contact information

### Enforcement

We enforce license violations through:
1. **Friendly Contact**: Initial outreach to resolve amicably
2. **Cease and Desist**: Formal legal notice if needed
3. **DMCA Takedown**: For hosted violations (GitHub, etc.)
4. **Litigation**: Last resort for significant violations

---

## Frequently Asked Questions

### Can I use NexusZero Protocol in my commercial product?

**Yes!** The MIT License explicitly allows commercial use without any fees or royalties.

### Do I need to open source my modifications?

**No.** The MIT License does not require you to share your modifications. However, we encourage you to contribute improvements back to benefit the community.

### Can I sell NexusZero Protocol?

**Yes.** You can sell products that include or are based on NexusZero Protocol. You just need to include the MIT License and copyright notice.

### Do I need to pay for a commercial license?

**For Core Protocol: No.** The core protocol is free under MIT License.

**For Premium Features: Maybe.** If you want access to premium features, enterprise support, or proprietary modules, you'll need a commercial license.

### Can I remove the copyright notice?

**No.** The MIT License requires you to include the copyright notice in all copies or substantial portions of the software.

### What if I want to use a GPL library with NexusZero?

**Be Careful.** Using GPL libraries in conjunction with NexusZero Protocol may create license compatibility issues. Consult with a lawyer or contact us for guidance.

### Can I fork NexusZero and create a competing project?

**Yes.** The MIT License allows you to fork and create competing projects. However:
- You cannot use our trademarks (NexusZero™) in your project name
- You must include our copyright notice and MIT License
- Our patents still apply (consider licensing if you're implementing patented features)

### What happens to my contributions?

Your contributions are licensed under the MIT License (same as the project). By signing the CLA, you grant us the right to use your contributions commercially and potentially dual-license them.

### Can I get an exception to the license terms?

**Contact Us.** If you have specific needs that aren't met by the MIT License or our commercial licenses, email legal@nexuszero.io to discuss custom licensing arrangements.

---

## License Changes

### Will the license ever change?

**Core Protocol**: We commit to keeping the core protocol under MIT License (or a similarly permissive license) permanently. This ensures long-term stability for users.

**Premium Features**: Premium features may be under different licenses (commercial licenses).

**New Modules**: Future modules may be dual-licensed (MIT + Commercial) from the start.

### How will users be notified of license changes?

- 90-day advance notice via email and GitHub
- Prominent notification on repository and website
- Existing versions remain under their original license
- Users can choose to stay on the old license or upgrade

---

## Resources

### License Texts

- [MIT License](https://opensource.org/licenses/MIT)
- [Apache License 2.0](https://opensource.org/licenses/Apache-2.0)
- [BSD Licenses](https://opensource.org/licenses/BSD-3-Clause)

### License Choosers

- [Choose A License](https://choosealicense.com/)
- [SPDX License List](https://spdx.org/licenses/)
- [OSI Approved Licenses](https://opensource.org/licenses)

### Legal Resources

- [GitHub's Open Source Guide](https://opensource.guide/legal/)
- [Software Freedom Law Center](https://softwarefreedom.org/)
- [Open Source Initiative](https://opensource.org/)

---

## Contact

**General Licensing Questions**: legal@nexuszero.io  
**Commercial Licensing**: enterprise@nexuszero.io  
**Trademark Questions**: trademarks@nexuszero.io  
**License Violations**: legal@nexuszero.io

---

**Document Version**: 1.0  
**Last Updated**: November 23, 2025  
**Next Review**: Annually or upon significant changes
