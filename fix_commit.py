import subprocess
import re

# Get commit message
result = subprocess.run(['git', 'log', '--format=%B', '-n', '1', 'f6d8846f9c1df36e769ec6aaa470df18582a6701'], capture_output=True, text=True)
msg = result.stdout

# Remove Co-Authored-By lines
new_msg = re.sub(r'Co-Authored-By: Claude Opus.*?\n', '', msg)

with open('msg.txt', 'w') as f:
    f.write(new_msg.strip() + '\n')
