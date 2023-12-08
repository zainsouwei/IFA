import os
import re

# Function to read content from a page
def read_page_content(page_path):
    with open(page_path, 'r') as file:
        return file.read()

# Function to extract unchecked tasks from a section
def extract_tasks(section_content):
    return re.findall(r'\* \[ \] (.+)', section_content)

def extract_bullet_points(section_content):
    # Use a regular expression to capture bullet points with indentation
    return re.findall(r'^(\s*\*\s+.+)$', section_content, flags=re.MULTILINE)


def extract_section(markdown_content, section_header,section_ender=""):
    # Construct the regular expression pattern
    pattern = r'{}(.*?)(?=\n##{}|$)'.format(section_header, section_ender)
    
    # Use re.search to find the section
    section_match = re.search(pattern, markdown_content, re.DOTALL)
    
    if section_match:
        # Use group(1) to get the content captured by the (.*?) group
        return section_match.group(1)
    else:
        return None

# Function to update the code-qt.md sheet
def update_code_qt_sheet():
    # Define paths with raw strings
    directory = r"C:\Users\zaisou\Desktop\ICASPADE\Documentation"
    code_qt_path = os.path.join(directory, 'code-qt.md')
    algorithm_qt_path = os.path.join(directory, 'algorithm-qt.md')
    code_sum_path = os.path.join(directory, 'code-summary.md')
    algorithm_sum_path = os.path.join(directory, 'algorithm-summary.md')

    allcontent = []
    with open(code_qt_path, 'r') as file:
        code_qt_content = file.read()
        allcontent.append(code_qt_content)

    with open(algorithm_qt_path, 'r') as file:
        algorithm_qt_content = file.read()
        allcontent.append(algorithm_qt_content)
    
    with open(code_sum_path, 'r') as file:
        code_sum_content = file.read()
        allcontent.append(code_sum_content)
    
    with open(algorithm_sum_path, 'r') as file:
        algorithm_sum_content = file.read()
        allcontent.append(algorithm_sum_content)

    # Iterate through each page in the directory
    for page_name in os.listdir(directory):
        if page_name.endswith(".md") and re.search(r'\d{4}\.\d{2}\.\d{2}', page_name):
            # Construct the full path to the page
            page_path = os.path.join(directory, page_name)

            # Read content from the page
            page_content = read_page_content(page_path)
            sections = [r'##\s*Algorithm',r'##\s*Code']
            for i, section in enumerate(sections):
                section_content = extract_section(page_content, section,r'[^#]')
                subsections = [r'###\s*Developments',r'###\s*Questions',r'###\s*ToDo']
                for j, subsection in enumerate(subsections):
                    subsection_block = extract_section(section_content, subsection, r'#')
                    if subsection_block:
                        if j == 0:
                            if i == 0:
                                targetbulletpage = 3
                            else:
                               targetbulletpage = 3
                            title =  re.search(r'#(.*?)(?=\n)', page_content, re.DOTALL).group(1)
                            if f'{title}' not in allcontent[targetbulletpage]:
                                allcontent[targetbulletpage] +=  f'\n##{title}'
                                allcontent[targetbulletpage] +=  f'{subsection_block}'
                        else:
                            unchecked = extract_tasks(subsection_block)
                            # alg_q_content
                            if i == 0 and j == 1:
                                target_qt_content = extract_section(algorithm_qt_content, r'##\s*Questions',r'[^#]')
                            # alg_t_content
                            elif i == 0 and j == 2:
                                target_qt_content = extract_section(algorithm_qt_content, r'##\s*ToDo',r'[^#]')
                            # code_q_content
                            elif i == 1 and j == 1:
                                target_qt_content = extract_section(code_qt_content, r'##\s*Questions',r'[^#]')
                            # code_t_content
                            elif i == 1 and j == 2:
                                target_qt_content = extract_section(code_qt_content, r'##\s*ToDo',r'[^#]')
                            
                            for question in unchecked:
                                # Check if the question is already in the code-qt.md sheet
                                if f'* [ ] {question}' not in target_qt_content:
                                    # Add the unchecked question to code-qt.md
                                     target_qt_content += f'* [ ] {question}\n'

                #             # Mark checked tasks in the original page
                #             checked_code_questions = re.findall(r'\* \[x\] (.+)', page_content)
                #             for question in checked_code_questions:
                #                 # Check if the question is in the code-qt.md sheet
                #                 if f'* [ ] {question}' in code_qt_content:
                #                     # Remove the checked question from code-qt.md
                #                     code_qt_content = code_qt_content.replace(f'* [ ] {question}\n', '')
                #                 # Mark the question as checked in the original page
                #                 page_content = page_content.replace(f'* [ ] {question}', f'* [x] {question}')

                # # Write the updated content back to the original page
                # with open(page_path, 'w') as file:
                #     file.write(page_content)

    with open(code_qt_path, 'w') as file:
        file.write(code_qt_content)
    with open(algorithm_qt_path, 'w') as file:
        file.write(algorithm_qt_content)
    with open(code_sum_path, 'w') as file:
        file.write(allcontent[2])
    with open(algorithm_sum_path, 'w') as file:
        file.write(allcontent[3])

# Run the update_code_qt_sheet function
update_code_qt_sheet()