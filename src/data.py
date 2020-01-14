"""Module for loading plaintext data"""

class DataLoader:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.text = f.read()  
        self.get_lines

    """removes execution metatdata from plaintext"""
    def clean_agiga_text(self):
        self.text = " ".join(self.get_lines()[13:-7])

    def get_text(self):
        return self.text

    def get_lines(self):
        lines = self.text.split("\n")
        self.lines = lines
        return self.lines

    def sanity_check(self):
        print(self.get_text()[:200], self.get_text()[-200:])

    def __len__(self):
        return len(self.lines)

def build_loader(file_path):
    loader = DataLoader(file_path)
    loader.clean_agiga_text()
    return loader

def export_events(document, file_path):
    with open(file_path, "w") as file:
        for event in document.left_events:
            file.write("\n")
            file.write(str(event))
            file.write(" ")
            file.write(str(document.events[event]))

        for event in document.right_events:
            file.write("\n")
            file.write(str(event))
            file.write(" ")
            file.write(str(document.events[event]))