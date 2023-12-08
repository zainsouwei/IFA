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
    code_qt_path = os.path.join(directory, 'code-todo.md')
    algorithm_qt_path = os.path.join(directory, 'algorithm-questions.md')
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
                                targetsum = 3
                            else:
                               targetsum = 3
                            title =  re.search(r'#(.*?)(?=\n)', page_content, re.DOTALL).group(1)
                            if f'{title}' not in allcontent[targetsum]:
                                allcontent[targetsum] +=  f'\n##{title}'
                                allcontent[targetsum] +=  f'{subsection_block}'
                        else:
                            # alg_q_content
                            if i == 0:
                                targetsum = 1
                            # code_q_content
                            elif i == 1:
                                targetsum = 0
                            
                            # Add unchecked to the summary sheet
                            unchecked = extract_tasks(subsection_block)
                            for question in unchecked:
                                # Check if that the question is not in the summary sheet
                                if f'{question}' not in allcontent[targetsum]:
                                    # Add the unchecked question to code-qt.md
                                     allcontent[targetsum] += f'* [ ] {question}\n'
                            
                            # In case I checked the question in the original page, remove it from the summary sheet 
                            checked_questions = re.findall(r'\* \[x\] (.+)', subsection_block)
                            for question in checked_questions:
                                if f'* [ ] {question}' in allcontent[targetsum]:
                                    allcontent[targetsum] = allcontent[targetsum].replace(f'* [ ] {question}\n', '')
    
    # Checked Questions in the Summary Sheets
    for i in range(0,2):
        checked_questions_sum = re.findall(r'\* \[x\] (.+)', allcontent[i])
        
        # Remove from summary sheet
        for question in checked_questions_sum:
                if f'* [x] {question}' in allcontent[i]:
                    allcontent[i] = allcontent[i].replace(f'* [x] {question}\n', '')
        
        # Mark as done in original page
        for page_name in os.listdir(directory):
            if page_name.endswith(".md") and re.search(r'\d{4}\.\d{2}\.\d{2}', page_name):
                # Construct the full path to the page
                page_path = os.path.join(directory, page_name)
                # Read content from the page
                page_content = read_page_content(page_path)
                
                # Find the checked questions in the page and mark them as checked 
                for question in checked_questions_sum:
                    if f'* [ ] {question}' in page_content:
                        page_content = page_content.replace(f'* [ ] {question}\n', f'* [x] {question}\n')
            
            with open(page_path, 'w') as file:
                file.write(page_content)

    with open(code_qt_path, 'w') as file:
        file.write(allcontent[0])
    with open(algorithm_qt_path, 'w') as file:
        file.write(allcontent[1])
    with open(code_sum_path, 'w') as file:
        file.write(allcontent[2])
    with open(algorithm_sum_path, 'w') as file:
        file.write(allcontent[3])

# Run the update_code_qt_sheet function
update_code_qt_sheet()