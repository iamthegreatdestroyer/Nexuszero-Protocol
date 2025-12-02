# Copyright Header Templates

Use these templates in all source code files to establish copyright protection.

---

## Rust (.rs files)

```rust
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Steve (iamthegreatdestroyer)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
```

---

## Python (.py files)

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Steve (iamthegreatdestroyer)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
```

---

## TypeScript/JavaScript (.ts, .tsx, .js files)

```typescript
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Steve (iamthegreatdestroyer)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
```

---

## Markdown (.md files)

```markdown
<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Steve (iamthegreatdestroyer)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->
```

---

## HTML (.html files)

```html
<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Steve (iamthegreatdestroyer)
-->
```

---

## CSS (.css, .scss files)

```css
/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Steve (iamthegreatdestroyer)
 */
```

---

## Shell Scripts (.sh, .bash files)

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Steve (iamthegreatdestroyer)
```

---

## PowerShell (.ps1 files)

```powershell
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Steve (iamthegreatdestroyer)
```

---

## YAML (.yml, .yaml files)

```yaml
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Steve (iamthegreatdestroyer)
```

---

## TOML (.toml files)

```toml
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Steve (iamthegreatdestroyer)
```

---

## JSON (.json files)

JSON doesn't support comments, so include copyright in a `"_copyright"` field:

```json
{
  "_copyright": "Copyright (c) 2025 Steve (iamthegreatdestroyer). SPDX-License-Identifier: MIT",
  ...
}
```

---

## Short Form Header (for small files)

For short utility files or scripts:

```
// Copyright (c) 2025 Steve (iamthegreatdestroyer)
// SPDX-License-Identifier: MIT
```

---

## Proprietary/Confidential Header (for trade secrets)

For files containing trade secrets that should NOT be open source:

```rust
// PROPRIETARY AND CONFIDENTIAL
// Copyright (c) 2025 NexusZero Protocol
// All Rights Reserved.
//
// This file contains trade secrets and confidential information
// of NexusZero Protocol. Unauthorized disclosure is prohibited.
//
// Access restricted to authorized personnel only.
// Distribution, copying, or use without explicit written permission is prohibited.
```

---

## Patent-Pending Header (for patent-pending innovations)

For files implementing patent-pending innovations:

```rust
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Steve (iamthegreatdestroyer)
//
// PATENT PENDING: This file implements technology described in
// patent application(s): [Patent Application Number(s)]
//
// [Standard MIT License text follows...]
```

---

## Third-Party Attribution Header

When incorporating third-party code:

```rust
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Steve (iamthegreatdestroyer)
//
// Portions derived from [Original Project Name]
// Copyright (c) [Year] [Original Author]
// Licensed under [Original License]
// Source: [URL]
//
// Modifications:
// - [Description of modifications]
// - [Date and author of modifications]
```

---

## Usage Guidelines

### When to Add Headers

**Required**:
- All source code files (.rs, .py, .ts, .js, etc.)
- All documentation files (.md)
- All configuration files
- All scripts

**Optional**:
- Test files (recommended but not required)
- Build artifacts
- Generated files (add note that file is generated)

### How to Add Headers

1. **New Files**: Add header as first lines of file
2. **Existing Files**: Add header before any code/content
3. **Generated Files**: Add comment indicating file is generated
4. **Automation**: Use pre-commit hooks to validate headers

### Automated Header Checking

Use tools to enforce headers:

**Rust**:
```bash
# Check for copyright headers
cargo install cargo-about
cargo about generate about.hbs > THIRD_PARTY_LICENSES.html
```

**Python**:
```bash
# Check for copyright headers
pip install licenseheaders
licenseheaders -t COPYRIGHT_HEADER_TEMPLATE.txt -d src/
```

**Node.js**:
```bash
# Check for copyright headers
npm install --save-dev license-checker
npx license-checker --summary
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Check for copyright headers in staged files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(rs|py|ts|js|md)$'); do
    if ! grep -q "Copyright (c) 2025 NexusZero Protocol" "$file"; then
        echo "ERROR: Missing copyright header in $file"
        exit 1
    fi
done

echo "All files have proper copyright headers"
exit 0
```

---

## SPDX License Identifiers

SPDX (Software Package Data Exchange) identifiers provide a standard way to indicate licenses:

- **MIT**: `SPDX-License-Identifier: MIT`
- **Apache-2.0**: `SPDX-License-Identifier: Apache-2.0`
- **GPL-3.0**: `SPDX-License-Identifier: GPL-3.0-only`
- **Proprietary**: `SPDX-License-Identifier: PROPRIETARY`

Benefits:
- Machine-readable
- Standardized across projects
- Supported by license scanners
- Reduces ambiguity

---

## Year Updates

**Question**: Should we update the year annually?

**Answer**: Two approaches:

1. **Range**: `Copyright (c) 2025-2026 Steve (iamthegreatdestroyer)`
   - Update annually or when file is modified
   - Clearly shows lifetime of modifications

2. **Initial Year Only**: `Copyright (c) 2025 Steve (iamthegreatdestroyer)`
   - Simpler, doesn't require updates
   - Still valid copyright protection

**Recommendation**: Use initial year only to avoid maintenance burden.

---

## Legal Review

Copyright headers should be reviewed by legal counsel to ensure:
- Proper copyright notice
- Correct license identifier
- Appropriate attribution
- Compliance with jurisdictional requirements

**Last Legal Review**: [To be scheduled]  
**Next Review**: [Annually or when licensing changes]

---

## Questions?

Contact: legal@nexuszero.io

---

**Template Version**: 1.0  
**Last Updated**: November 23, 2025
