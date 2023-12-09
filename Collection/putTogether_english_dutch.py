# Open the first file for reading with UTF-8 encoding
with open("dutchText.txt", "r", encoding="utf-8") as file1:
    # Open the second file for reading with UTF-8 encoding
    with open("English_collected_sentences.txt", "r", encoding="utf-8") as file2:
        # Open the third file for writing with UTF-8 encoding
        with open("Data.txt", "w", encoding="utf-8") as file3:
            # Initialize line numbers
            line_number = 1

            # Read lines from both files simultaneously in a loop
            while True:
                # Read a line from the first file
                line1 = file1.readline().strip()

                # Read a line from the second file
                line2 = file2.readline().strip()

                # Check if both files have reached the end
                if not line1 and not line2:
                    break

                # Write the lines to the third file with line numbers
                if line1:
                    file3.write(f"nl|{line1}\n")
                if line2:
                    file3.write(f"en|{line2}\n")

                # Increment the line number
                line_number += 1
