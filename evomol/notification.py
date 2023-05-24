class Observer:

    def __init__(self, observable):
        observable.subscribe(self)

    def update(self, *args, **kwargs):
        pass


class Observable:

    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for obs in self._observers:
            obs.update(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self._observers.remove(observer)