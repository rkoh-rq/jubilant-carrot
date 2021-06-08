class EarlyStopping():
    def __init__(self, patience = 10, min_loss = 0.5, hit_min_before_stopping = False):
        self.patience = patience
        self.counter = 0
        self.hit_min_before_stopping = hit_min_before_stopping
        if hit_min_before_stopping:
            self.min_loss = min_loss
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                if self.hit_min_before_stopping == True and loss > self.min_loss:
                    print("Cannot hit min loss, will continue")
                    self.counter -= self.patience
            else:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0