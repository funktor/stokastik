import re
from xcrf.preprocessor.preprocessor import Preprocessor
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.NEWTAGS = []
        self.NEWATTRS = []
        self._lines = []

    def handle_data(self, data):
        self._lines.append(data)

    def read(self, data):
        # clear the current output before re-use
        self._lines = []
        # re-set the parser's state before re-use
        self.reset()
        self.feed(data)
        return ' '.join(self._lines)


class HTMLPreprocessor(Preprocessor):
    def __init__(self):
        Preprocessor.__init__(self)
        self.parser = MyHTMLParser()

    def preprocess(self, data):
        data = self.parser.read(data)
        return data

    def preprocess_all(self, data):
        data = [self.parser.read(d) for d in data]
        return data


if __name__ == "__main__":
    html_prep = HTMLPreprocessor()

    data = "<b>Phoenix Suns BBQ Grill Travel Tools Set Features:</b><br/><ul><li>Product Type: Officially Licensed Phoenix Suns BBQ Grill Travel AccessoriesTools Set</li><li>Material: Polyester Canvas-Metal Grill Tools</li><li>Additional Info: Includes: 1 Spatula, 1 Fork, 1 Tongs, 1 Tote</li><li>Condition: New</li><li>Officially Licensed with Free Shipping</li></ul>"

    print(html_prep.preprocess(data))
