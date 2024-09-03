class Observer:

    def __init__(self):
        pass

    def update(self, *args, **kwargs):
        pass


class Observable:

    def __init__(self):
        self.observers = []

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for obs in self.observers:
            obs.update(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self.observers.remove(observer)