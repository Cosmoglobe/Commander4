
class CompSepDataset:
    def __init__(self):
        self.map = None
        self.noise = None
        self.beam = None
        self.bandpass = None

    def apply_beam(self, input):
        return self.beam*input
        
    
