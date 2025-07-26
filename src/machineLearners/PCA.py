import numpy as np

class PCA():
    def __init__(self, n_components, center_data=True):
        self.n_components = n_components
        self.center_data=center_data
        self.mean = False
        self.transformationMatrix=False
        self.G = False

    def center(self, X):
        bar = ((1/np.size(X,0)) * np.transpose(X) @ np.ones(np.size(X,0)))[:,np.newaxis]
        tilda=(X+np.transpose(bar @ np.ones((1,np.size(X,0)))))
        self.mean = bar
        return tilda

    def train(self,X):
        data = X
        if(self.center_data):
            data = self.center(data)
        U, s, V = np.linalg.svd(data, full_matrices=False)
        G = V[0:self.n_components]
        self.transformationMatrix = np.transpose(G)
        return np.transpose(G @ np.transpose(data))
    
    def test(self,X):
        data = X
        if(self.center_data):
            data=(X+np.transpose(self.mean @ np.ones((1,np.size(X,0)))))
        return np.transpose(np.transpose(self.transformationMatrix) @ np.transpose(data))
    
    def reconstruct(self, Xp):
        return np.transpose(self.transformationMatrix @ np.transpose(Xp))
    
    def reconstruction_error(self, A, B):
        data = A
        if(self.center_data):
            bar = ((1/np.size(A,0)) * np.transpose(A) @ np.ones(np.size(A,0)))[:,np.newaxis]
            tilda=(A+np.transpose(bar @ np.ones((1,np.size(A,0)))))
            data=tilda
        AminusB = data-B
        return np.linalg.norm(AminusB)**2