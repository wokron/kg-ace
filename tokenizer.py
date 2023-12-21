from typing import List
from flair.data import SegtokTokenizer, Tokenizer


#将文本进行分词处理
class FlairBertTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = SegtokTokenizer()

    def tokenize(self, text: str) -> List[str]:
        tok_list = self.tokenizer.tokenize(text)
        result = []
        i = 0
        while i < len(tok_list):
            if tok_list[i] == "[":
                result.append("".join(tok_list[i : i + 3]))
                i += 2
            else:
                result.append(tok_list[i])
            i += 1
        return result
