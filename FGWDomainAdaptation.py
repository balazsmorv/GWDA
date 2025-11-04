import torch
import ot


class FGWDomainAdaptation:

    cost_matrix = None
    P = None
    random_state = 42

    def __init__(self, Xs, Xt):
        self.Xs = Xs
        self.Xt = Xt

    def compute_distances(self):
        self.cost_matrix = torch.zeros((len(self.Xs), len(self.Xt)))

        total = len(self.Xs) * len(self.Xt)
        done = 0
        for i, source_graph in enumerate(self.Xs):
            for j, target_graph in enumerate(self.Xt):
                self.cost_matrix[i, j] = ot.gromov.gromov_wasserstein2(C1=source_graph.x,
                                                            C2=target_graph.x,
                                                            p=torch.ones(200) / 200,
                                                            q=torch.ones(200) / 200,
                                                            loss_fun='square_loss',
                                                            symmetric=None,
                                                            log=False,
                                                            armijo=False,
                                                            G0=None,
                                                            max_iter=100,
                                                            tol_rel=1e-9,
                                                            tol_abs=1e-9)
                done += 1
                print(f"{done}/{total}")


        self.P = ot.emd(a=torch.ones(len(self.Xs)) / len(self.Xs),
                        b=torch.ones(len(self.Xt)) / len(self.Xt),
                        M=self.cost_matrix,
                        numItermax=100000)

    def transport(self):
        transp_Xt = []
        for i in range(len(self.Xs)):
            indices = torch.nonzero(self.P[i], as_tuple=False).flatten().tolist()
            print(indices)
            barycenter = ot.gromov.gromov_barycenters(N=200,
                                                     Cs=[self.Xt[idx].x for idx in indices],
                                                     ps=None, # uniform
                                                     p=None, # uniform
                                                     lambdas=None, # uniform
                                                     loss_fun='square_loss',
                                                     symmetric=True,
                                                     armijo=False,
                                                     max_iter=10,
                                                     tol=1e-4,
                                                     stop_criterion='barycenter',
                                                     warmstartT=False,
                                                     verbose=True,
                                                     log=False,
                                                     init_C=None,
                                                     random_state=self.random_state)
            transp_Xt.append(barycenter)
            print(f"Done {i}/{len(self.Xs)}")
        return transp_Xt