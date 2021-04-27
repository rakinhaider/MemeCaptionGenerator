import progressbar


class MyProgressBar():
    def __init__(self, total):
        widgets = ['Loading',
                   progressbar.Bar(), progressbar.Percentage()]
        self.bar = progressbar.ProgressBar(maxval=total,
                                           widgets=widgets)
        self.bar.start()

    def update(self, val):
        self.bar.update(self.bar.currval + val)

    def finish(self):
        self.bar.finish()