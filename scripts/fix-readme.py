#!/usr/bin/env python3
"""Fix README.md formatting by converting escaped newlines and single-backtick code fences."""

import re

# Read the file
with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace escaped newlines with actual newlines
content = content.replace('\\n', '\n')

# Replace single-backtick code fences with triple-backtick fences
# Match patterns like `powershell, `bash, `python, `text, and standalone `
content = re.sub(r'^`([a-z]+)$', r'```\1', content, flags=re.MULTILINE)
content = re.sub(r'^`$', r'```', content, flags=re.MULTILINE)

# Write the fixed content back
with open('README.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("✅ README.md has been fixed!")
print("   - Converted escaped \\n to actual newlines")
print("   - Converted single-backtick fences to triple-backtick fences")
