import os
import re

# Function to read content from a page
def read_page_content(page_path):
    with open(page_path, 'r') as file:
        return file.read()

# Function to extract unchecked tasks from a section
def extract_tasks(section_content):
    return re.findall(r'\* \[ \] (.+)', section_content)

# Function to update the code-qt.md sheet
def update_code_qt_sheet():
    # Define paths
    script_directory = r"C:\Users\zaisou\Desktop\ICASPADE\Documentation"
    pages_directory = r"C:\Users\zaisou\Desktop\ICASPADE\Documentation"
    code_qt_path = os.path.join(script_directory, 'code-qt.md')

    # Read content from the code-qt.md sheet
    with open(code_qt_path, 'r') as file:
        code_qt_content = file.read()

    # List to store checked tasks
    checked_tasks = []

    # Iterate through each page in the directory
    for page_name in os.listdir(pages_directory):
        if page_name.endswith(".md"):
            # Construct the full path to the page
            page_path = os.path.join(pages_directory, page_name)

            # Read content from the page
            page_content = read_page_content(page_path)

            # Extract unchecked code questions and todos
            code_section = re.search(r'## Code(.+?)##', page_content, re.DOTALL)
            if code_section:
                unchecked_code_questions = extract_tasks(code_section.group(1))
                for question in unchecked_code_questions:
                    # Check if the question is already in the code-qt.md sheet
                    if f'* [ ] {question}' not in code_qt_content:
                        # Add the unchecked question to code-qt.md
                        code_qt_content += f'* [ ] {question}\n'

            # Mark checked tasks in the original page
            checked_code_questions = re.findall(r'\* \[x\] (.+)', page_content)
            for question in checked_code_questions:
                # Check if the question is in the code-qt.md sheet
                if f'* [ ] {question}' in code_qt_content:
                    # Remove the checked question from code-qt.md
                    code_qt_content = code_qt_content.replace(f'* [ ] {question}\n', '')
                # Mark the question as checked in the original page
                page_content = page_content.replace(f'* [ ] {question}', f'* [x] {question}')

            # Write the updated content back to the original page
            with open(page_path, 'w') as file:
                file.write(page_content)

    # Write the updated content back to the code-qt.md sheet
    with open(code_qt_path, 'w') as file:
        file.write(code_qt_content)

# Run the update_code_qt_sheet function
update_code_qt_sheet()