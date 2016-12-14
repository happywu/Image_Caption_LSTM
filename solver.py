class Solver:

    def step(self, batch, model, cost_function, **kwargs):

        cg = cost_function(batch, model)
        cost = cg['cost']
        grads = cg['grads']
        stats = cg['stats']

        for p in update:
            if p in grads:
                dx = - learning_rate * grads[p]
                model[p] += dx

        out = {}
        out['cost'] = cost
        out['stats'] = stats

        return out

