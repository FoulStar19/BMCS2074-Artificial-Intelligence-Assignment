import os
import glob

def delete_label_lines(directory_path, label_to_delete, file_pattern="*.txt"):
    """
    Delete entire lines containing a specific class label in YOLO format annotation files.
    
    Args:
        directory_path: Path to the directory containing the annotation files
        label_to_delete: The class number to delete (e.g., 0, 1, 2, etc.)
        file_pattern: Pattern to match files (default: "*.txt")
    """
    
    # Get all txt files in the directory
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not file_paths:
        print(f"No files found matching pattern '{file_pattern}' in '{directory_path}'")
        return
    
    modified_count = 0
    total_lines_deleted = 0
    
    for file_path in file_paths:
        try:
            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Keep lines that don't have the label to delete
            new_lines = []
            lines_deleted = 0
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    new_lines.append(line)
                    continue
                
                # Split the line into parts
                parts = line.strip().split()
                
                # Check if the first part matches the label to delete
                if parts and parts[0] == str(label_to_delete):
                    # Skip this line (delete it)
                    lines_deleted += 1
                else:
                    # Keep this line
                    new_lines.append(line)
            
            # Write back to file if lines were deleted
            if lines_deleted > 0:
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)
                modified_count += 1
                total_lines_deleted += lines_deleted
                print(f"Modified: {os.path.basename(file_path)} - Deleted {lines_deleted} lines")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nTotal files modified: {modified_count} out of {len(file_paths)}")
    print(f"Total lines deleted: {total_lines_deleted}")

def main():
    # Example usage - modify these variables as needed
    directory = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset\labels"  # Current directory, change to your actual directory path
    label_to_delete = 2 # Change this to the label number you want to delete
    
    print(f"Deleting all lines with label {label_to_delete}...")
    delete_label_lines(directory, label_to_delete)

if __name__ == "__main__":
    main()