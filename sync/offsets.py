import os


class Offsets(dict):
    def __init__(self, *args, **kwargs):
        self.base_key = kwargs.pop("base", None)
        super().__init__(*args, **kwargs)
        if self.base_key is None and len(self) > 0:
            self.base_key = list(self.keys())[0]
        if len(self) > 0:
            self.balance_base()

    def balance_base(self):
        base_value = self[self.base_key]
        if base_value != 0:
            for key in self.keys():
                self.__setitem__(key, self[key]-base_value, False)

    def __setitem__(self, k, v, balance=True) -> None:
        super().__setitem__(k, v)
        if balance:
            self.balance_base()

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + "base_key:" + self.base_key

    def to_file(self, file_path: str):
        with open(os.path.join(file_path, "offsets.txt"), "w") as f:
            f.write(self.__repr__())
    
    @staticmethod
    def from_file(file_path: str):
        with open(os.path.join(file_path, "offsets.txt"), "r") as f:
            return Offsets(eval(f.readline()), base=f.readline().replace('\n','').split(':')[1])

    @staticmethod
    def verify_file(file_path: str):
        p = os.path.join(file_path, "offsets.txt")
        if os.path.exists(p):
            try:
                Offsets.from_file(file_path)
            except Exception as e:
                return False
            return True
        return False
