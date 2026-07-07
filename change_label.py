import os
import glob

def change_label_in_files(directory_path, original_label, new_label, file_pattern="*.txt"):
    """
    Change the first number (class label) in YOLO format annotation files.
    
    Args:
        directory_path: Path to the directory containing the annotation files
        original_label: The original class number to change (e.g., 0, 1, 2, etc.)
        new_label: The new class number to replace with (e.g., 99 for temporary label)
        file_pattern: Pattern to match files (default: "*.txt")
    """
    
    # Get all txt files in the directory
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not file_paths:
        print(f"No files found matching pattern '{file_pattern}' in '{directory_path}'")
        return
    
    modified_count = 0
    
    for file_path in file_paths:
        try:
            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Process each line
            new_lines = []
            file_modified = False
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    new_lines.append(line)
                    continue
                
                # Split the line into parts
                parts = line.strip().split()
                
                # Check if the first part matches the original label
                if parts and parts[0] == str(original_label):
                    # Replace with new label
                    parts[0] = str(new_label)
                    new_line = ' '.join(parts) + '\n'
                    new_lines.append(new_line)
                    file_modified = True
                else:
                    new_lines.append(line)
            
            # Write back to file if modified
            if file_modified:
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)
                modified_count += 1
                print(f"Modified: {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nTotal files modified: {modified_count} out of {len(file_paths)}")

def main():
    # Example usage - modify these variables as needed
    directory = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset\labels"  # Current directory, change to your actual directory path
    original_label = 9  # Change this to your original label number
    new_label = 3      # Change this to your temporary label number
    
    print(f"Changing label {original_label} to {new_label} in all annotation files...")
    change_label_in_files(directory, original_label, new_label)

if __name__ == "__main__":
    main()
    