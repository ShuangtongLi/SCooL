import numpy as np
from scipy.special import psi
import random
from tqdm import tqdm
import util
#from formatted_logger import formatted_logger

#log = formatted_logger('MMSB', 'info')

class SBM:

    def __init__(self, Y, K, alpha = 1, clusterAssign = None, scale = 1.):
        """ follows the notations in the original NIPS paper
        :param Y: node by node interaction matrix, row=sources, col=destinations
        :param K: number of mixtures
        :param alpha: Dirichlet parameter
        :param rho: sparsity parameter
        """
        self.N = int(Y.shape[0])    # number of nodes
        self.K = K
        self.alpha = np.ones(self.K) 
        self.Y = Y

        self.optimize_rho = False
        self.max_iter = 2

        #variational parameters

        #self.gamma = np.random.dirichlet(self.alpha, size=self.N)
        #self.gamma = np.random.dirichlet([1]*self.K, size=self.N)

        gamma_base = 2 * 1.0 / float(self.K)
        gamma_max = gamma_base * 0.0
        gamma_min = gamma_base * 1.0
        self.gamma = np.random.random(size = (self.N, self.K)) * (gamma_max - gamma_min) + gamma_min


        self.phi = np.random.dirichlet(self.alpha, size=(self.N))
        for i in range(self.N):
            self.phi[i, :] = np.random.dirichlet(self.gamma[i])


        #self.B = np.random.random(size=(self.K,self.K))
        #self.B = np.ones((self.K, self.K))
        
        
        B = np.eye(self.K)*0.8
        self.B = B + np.ones([self.K, self.K])*0.2-np.eye(self.K)*0.2
        
        
        #self.B = np.eye(self.K)
        
        

        self.rho = (1.-np.sum(self.Y)/(self.N*self.N))  # 1 - density of matrix
        self.factor = 1e-6
        self.round = 1
        self.optimize_lr()
        self.clusterAssign = clusterAssign

        self.rho = 1.0 - float(np.sum(self.Y)) / float(self.N * self.N)
        self.acc_hist = []
        self.W = np.zeros((self.N, self.N))
        self.output_w = np.zeros((self.N, self.N))
        self.scale = scale


        #self.rho = 1
    def optimize_lr(self):
        '''
        self.round += 1
        K = 0.9
        self.lr = 1.0 / ((1.0 + self.round) ** K)
        '''
        '''
        self.round += 1
        self.lr = 0.1 * (0.9 ** self.round)
        '''

        self.lr = 0.05

    def variational_inference(self, converge_ll_fraction=1e-3):
        """ run variational inference for this model
        maximize the evidence variational lower bound
        :param converge_ll_fraction: terminate variational inference when the fractional change of the lower bound falls below this
        """
        converge = False
        old_ll = 0.
        iteration = 0

        for iteration in range(self.max_iter):
            
            for i in range(1):
                ll = self.run_e_step()
            #ll += self.run_m_step()
            self.optimize_lr()
            #self.showGamma(iteration)

            #print('W =')
            #print(self.output_W())
            #print("Y =")
            #print(self.Y)

            #np.savetxt("log/W_" + str(iteration) + ".txt", self.W, fmt = "%1.2f")
            #np.savetxt("log/W_output_" + str(iteration) + ".txt", self.output_W(), fmt = "%1.2f")
            #np.savetxt("log/Y.txt", self.Y, fmt = "%1.2f")

            
           

            #log.info('iteration %d, lower bound %.2f' %(iteration, ll))


    def run_e_step(self):
        """ compute variational expectations 
        """
        #print(self.B)
        for i in range(self.N):
            for j in range(self.N):
                self.optimize_W(i, j)
        ll = 0.
        i_list = list(range(self.N))
        j_list = list(range(self.N))
        random.shuffle(i_list)
        random.shuffle(j_list)
    
        for i in i_list:
            for iteration in range(1):
                self.optimize_phi(i)
            self.optimize_gamma(i)
        
        '''
        for i in i_list:
            for j in j_list:
                self.optimize_gamma(i, j)
                #self.optimize_B()
        '''
        



                
                
        '''
        for i in range(self.N):
            for g in range(self.K):
                self.gamma[i, g] = self.alpha[g] + np.sum(self.phi1[i, :, g]) + np.sum(self.phi2[:, i, g])
        '''
        #self.optimize_B()
        return ll

    def run_m_step(self):
        """ maximize the hyper parameters
        """
        ll = 0.

        self.optimize_alpha()
        #self.optimize_B()
        if self.optimize_rho:
            self.update_rho()

        return ll
    def init_phi(self, i):
        for k in range(self.K):
            self.phi[i, k] = 1.0 / float(self.K)
    def optimize_gamma(self, i):
        '''
        for g in range(self.K):
            self.gamma[i, g] = self.alpha[g] + np.sum(self.phi1[i, :, g]) + np.sum(self.phi2[:, i, g])
        for g in range(self.K):
            self.gamma[j, g] = self.alpha[g] + np.sum(self.phi1[j, :, g]) + np.sum(self.phi2[:, j, g])
        '''
        

        '''
        for g in range(self.K):
            tmp = self.alpha[g] + np.sum(self.phi1[i, :, g]) + np.sum(self.phi2[:, i, g])
            self.gamma[i, g] = (1 - self.lr) * tmp + self.lr * self.gamma[i, g]
        for g in range(self.K):
            tmp = self.alpha[g] + np.sum(self.phi1[j, :, g]) + np.sum(self.phi2[:, j, g])
            self.gamma[j, g]  = (1 - self.lr) * tmp + self.lr * self.gamma[j, g]
        '''


        '''
        for g in range(self.K):
            tmp = self.alpha[g] + np.sum(self.phi1[i, :, g]) + np.sum(self.phi2[:, i, g])
            gradient = tmp - self.gamma[i, g] - 0.01 * self.gamma[i, g]
            self.gamma[i, g] = self.gamma[i, g] + self.lr * gradient
        for g in range(self.K):
            tmp = self.alpha[g] + np.sum(self.phi1[j, :, g]) + np.sum(self.phi2[:, j, g])
            gradient = tmp - self.gamma[j, g] - 0.01 * self.gamma[i, g]
            self.gamma[j, g] = self.gamma[j, g] + self.lr * gradient
        '''

        for g in range(self.K):
            self.gamma[i, g] = self.alpha[g] + self.phi[i, g]
    def optimize_phi(self, i):
        
        '''
        fac1 = 3
        fac2 = 3
        new_phi1_ij = np.zeros(self.K)
        for k in range(self.K):
            tmp_phi = psi(self.gamma[j]+ self.factor)
            #tmp_phi = np.exp(tmp_phi)
            tmp_phi = tmp_phi / (np.sum(tmp_phi) + self.factor)
            for h in range(self.K):
                new_phi1_ij[k] += tmp_phi[h] * np.log(fac1 * (self.B[k,h] + self.factor)) * self.Y[i, j]
                new_phi1_ij[k] += tmp_phi[h] * np.log(fac1 * (1 - self.B[k,h] + self.factor)) * (1 - self.Y[i, j])
            new_phi1_ij[k] += psi(self.gamma[i, k] + self.factor)
            new_phi1_ij[k] -= psi(np.sum(self.gamma[i, :]) + self.factor)
        new_phi1_ij = np.exp(new_phi1_ij)
        new_phi1_ij = new_phi1_ij / (np.sum(new_phi1_ij) + self.factor)
        self.phi1[i, j, :] = new_phi1_ij



        new_phi2_ij = np.zeros(self.K)
        for k in range(self.K):
            tmp_phi = psi(self.gamma[i]+ self.factor)
            #tmp_phi = np.exp(tmp_phi)
            tmp_phi = tmp_phi / (np.sum(tmp_phi) + self.factor)
            for h in range(self.K):
                new_phi2_ij[k] += tmp_phi[h] * np.log(fac2 * (self.B[h,k] + self.factor)) * self.Y[i, j]
                new_phi2_ij[k] += tmp_phi[h] * np.log(fac2 * (1 - self.B[h,k] + self.factor)) * (1 - self.Y[i, j])
            new_phi2_ij[k] += psi(self.gamma[j, k] + self.factor)
            new_phi2_ij[k] -= psi(np.sum(self.gamma[j, :]) + self.factor)
        new_phi2_ij = np.exp(new_phi2_ij)
        new_phi2_ij = new_phi2_ij / (np.sum(new_phi2_ij) + self.factor)
        self.phi2[i, j, :] = new_phi2_ij
        '''
        
        


        '''
        fac1 = 20
        fac2 = 20
        new_phi1_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(fac1 * (self.B[k,h] + self.factor)) * self.Y[i, j]
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(fac1 * (1 - self.B[k,h] + self.factor)) * (1 - self.Y[i, j])
            new_phi1_ij[k] += psi(self.gamma[i, k] + self.factor)
            new_phi1_ij[k] -= psi(np.sum(self.gamma[i, :]) + self.factor)
        new_phi1_ij = np.exp(new_phi1_ij)
        new_phi1_ij = new_phi1_ij / (np.sum(new_phi1_ij) + self.factor)
        self.phi1[i, j, :] = new_phi1_ij



        new_phi2_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(fac2 * (self.B[h,k] + self.factor)) * self.Y[i, j]
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(fac2 * (1 - self.B[h,k] + self.factor)) * (1 - self.Y[i, j])
            new_phi2_ij[k] += psi(self.gamma[j, k] + self.factor)
            new_phi2_ij[k] -= psi(np.sum(self.gamma[j, :]) + self.factor)
        new_phi2_ij = np.exp(new_phi2_ij)
        new_phi2_ij = new_phi2_ij / (np.sum(new_phi2_ij) + self.factor)
        self.phi2[i, j, :] = new_phi2_ij
        '''
        
        
        

        '''        
        new_phi1_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(self.B[k,h] + self.factor) * self.Y[i, j]
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(1 - self.B[k,h] + self.factor) * (1 - self.Y[i, j])
            new_phi1_ij[k] += psi(self.gamma[i, k] + self.factor)
            new_phi1_ij[k] -= psi(np.sum(self.gamma[i, :]) + self.factor)
        new_phi1_ij = np.exp(new_phi1_ij)
        new_phi1_ij = new_phi1_ij / (np.max(new_phi1_ij) + self.factor)
        self.phi1[i, j, :] = new_phi1_ij




        new_phi2_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(self.B[h, k] + self.factor) * self.Y[i, j]
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(1 - self.B[h, k] + self.factor) * (1 - self.Y[i, j])
            new_phi2_ij[k] += psi(self.gamma[j, k] + self.factor)
            new_phi2_ij[k] -= psi(np.sum(self.gamma[j, :]) + self.factor)
        new_phi2_ij = np.exp(new_phi2_ij)
        new_phi2_ij = new_phi2_ij / (np.max(new_phi2_ij) + self.factor)
        self.phi2[i, j, :] = new_phi2_ij
        '''
        


        '''
        new_phi1_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(20 *(self.B[k,h] + self.factor)) * self.Y[i, j]
                new_phi1_ij[k] += self.phi2[i, j, h] * np.log(20 *(1 - self.B[k,h] + self.factor)) * (1 - self.Y[i, j])
            new_phi1_ij[k] += psi(self.gamma[i, k] + self.factor)
            new_phi1_ij[k] -= psi(np.sum(self.gamma[i, :]) + self.factor)
        new_phi1_ij_ind = np.argmax(new_phi1_ij)
        self.phi1[i, j, :] = np.zeros(self.K)
        self.phi1[i, j, new_phi1_ij_ind] = 1




        new_phi2_ij = np.zeros(self.K)
        for k in range(self.K):
            for h in range(self.K):
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(20 *(self.B[h, k] + self.factor)) * self.Y[i, j]
                new_phi2_ij[k] += self.phi1[i, j, h] * np.log(20 *(1 - self.B[h, k] + self.factor)) * (1 - self.Y[i, j])
            new_phi2_ij[k] += psi(self.gamma[j, k] + self.factor)
            new_phi2_ij[k] -= psi(np.sum(self.gamma[j, :]) + self.factor)
        new_phi2_ij_ind = np.argmax(new_phi2_ij)
        self.phi2[i, j, :] = np.zeros(self.K)
        self.phi2[i, j, new_phi2_ij_ind] = 1
        #print(self.phi1)
        '''
        scale = 1.0
        new_phi_ij = np.zeros(self.K)
        for k in range(self.K):
            for j in range(self.N):
                if j == i:
                    continue
                for h in range(self.K):
                    new_phi_ij[k] += self.phi[j, h] * np.log(scale*(self.B[k,h] + self.factor)) * self.W[i, j]
                    new_phi_ij[k] += self.phi[j, h] * np.log(scale*(self.B[h,k] + self.factor)) * self.W[j, i]
                    new_phi_ij[k] += self.phi[j, h] * np.log(scale*(1 - self.B[k,h] + self.factor)) * (1 - self.W[i, j])
                    new_phi_ij[k] += self.phi[j, h] * np.log(scale*(1 - self.B[h, k] + self.factor)) * (1 - self.W[j, i])
            new_phi_ij[k] += psi(self.gamma[i, k] + self.factor)
            new_phi_ij[k] -= psi(np.sum(self.gamma[i, :]) + self.factor)
        new_phi_ij = np.exp(new_phi_ij)
        new_phi_ij = new_phi_ij / (np.sum(new_phi_ij) + self.factor)
        self.phi[i, :] = new_phi_ij

        
        


        

    def optimize_alpha(self):
        return

    def optimize_B(self):
        for g in range(self.K):
            for h in range(self.K):
                tmp1=0.
                tmp2=0.
                for i in range(self.N):
                    for j in range(self.N):
                        tmp = self.phi[i, g] * self.phi[j, h]
                        tmp1 += tmp * self.Y[i, j]
                        tmp2 += tmp
                self.B[g,h] = tmp1/(tmp2+ self.factor) 
        return

    def update_rho(self):
        return
    def showGamma(self, i):
        gamma = self.gamma
        clusterAssign = self.clusterAssign
        gamma_mask = get_gamma_mask1(gamma)
        order, error = util.alignClusterAssignAndGamma(clusterAssign, gamma_mask)
        concated = np.concatenate((clusterAssign, gamma[:, order]), axis = 1)
        print(concated)
        np.savetxt("log/" + str(i) + ".txt", concated, fmt = "%1.2f")
        self.acc_hist.append(util.Accuracy(clusterAssign, gamma[:, order]))
        np.savetxt("log/" + "acc.txt", np.array(self.acc_hist), fmt = "%1.2f")
    def optimize_W(self, i, j):
        tmp = 0
        scale = self.scale
        for g in range(self.K):
            for h in range(self.K):
                tmp += self.phi[i, g] * self.phi[j, h] * np.log(self.B[g, h] + self.factor)
                tmp -= self.phi[i, g] * self.phi[j, h] * np.log( 1 - self.B[g, h] + self.factor)
        tmp = tmp #/ self.K / self.K
        tmp += self.Y[i, j]
        tmp = tmp / scale
        self.output_w[i, j] = tmp
        tmp = np.exp(-tmp) 
        #tmp += self.Y[i, j]# * 3
        self.W[i, j] = 1.0 / (1.0 + tmp)
        #print("w", self.W[i, j])
        #print(tmp)
        return
    def output_W(self):
        output_w = self.W
        return output_w / np.sum(output_w, axis = 1).reshape(-1, 1)


def generateMat1(n, k, sampleNum):
    import random
    clusterAssign = np.zeros((n, k))
    numList = [i for i in range(k)]
    for i in range(n):
        sampleList = random.sample(numList, sampleNum)
        for num in sampleList:
            clusterAssign[i][num] = 1.0
    print("clusterAssign:")
    print(clusterAssign)
    result = np.matmul(clusterAssign, clusterAssign.transpose())
    max_num = np.max(result)

    min_result = 0.5
    max_result = 1
    for i in range(len(result)):
        for j in range(len(result[i])):
            if max_num == 1:
                continue
            val = result[i][j]
            if val >= 1:
                post_val = (max_result - min_result) * float(val - 1) / float(max_num - 1) + min_result
                result[i][j] = post_val

    print("True adj mat:")
    print(result)
    return result, clusterAssign
def generateMat(n, k, sampleNum):
    import random
    clusterAssign = np.zeros((n, k))
    numList = [i for i in range(k)]
    for i in range(n):
        sampleList = random.sample(numList, sampleNum)
        for num in sampleList:
            clusterAssign[i][num] = 1.0
    print("clusterAssign:")
    print(clusterAssign)
    result = np.matmul(clusterAssign, clusterAssign.transpose())
    #result = result - np.min(result, axis = 1, keepdims = True)
    #result = result / np.max(result, axis = 1, keepdims = True)

    print("True adj mat:")
    print(result)
    return result, clusterAssign

def generateMatFromLog(path = "/ghome/list/StructralPrior/SingleTrainValidate/log/2022-06-08_08:55:23/TrainLoss/epoch"):
    LocalPath = path + "0.npy"
    mixingMat = np.load(LocalPath)
    print(mixingMat)

    factor = 1e-10
    Y = -mixingMat
    Y = Y - np.min(Y, axis = 1, keepdims = True)
    Y = Y / (np.max(Y, axis = 1, keepdims = True) + factor)
    print(Y)
    return Y

def generateMat2(n, k, sampleNum):
    import random
    clusterAssign = np.zeros((n, k))
    numList = [i for i in range(k)]
    for i in range(n):
        sampleList = random.sample(numList, sampleNum)
        for num in sampleList:
            clusterAssign[i][num] = 1.0
    print("clusterAssign:")
    print(clusterAssign)
    result = np.matmul(clusterAssign, clusterAssign.transpose())
    #result = result - np.min(result, axis = 1, keepdims = True)
    #result = result / np.max(result, axis = 1, keepdims = True)
    result = result >= 1
    result = result.astype(np.float32)
    print("True adj mat:")
    print(result)
    return result, clusterAssign

def get_gamma_mask1(gamma):
    x = np.argsort(gamma, axis = 1)
    x = x[:, -1:]
    mask = np.zeros((gamma.shape[0], gamma.shape[1]))
    for i in range(len(gamma)):
        for t in x[i]:
            mask[i][t] = 1
    return mask

def get_gamma_mask(gamma):
    x = np.max(gamma, axis = 1) * 0.5
    x = x.reshape((-2, 1))
    mask = gamma >= x
    mask = mask.astype(np.float32)
    return mask


def main():
    np.set_printoptions(precision=2, suppress = True)
    """ test with interaction matrix:
    1 1 0 0 0
    1 1 0 0 0 
    0 0 1 1 1
    0 0 1 1 1
    0 0 1 1 1
    """
    #Y = np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,1,1,1],[0,0,1,1,1],[0,0,1,1,1]])
    #Y = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,1],[1,0,0,0,1]])
    n = 15
    k = 3
    m = 1

    Y, clusterAssign = generateMat2(n, k, m)
    #Y = generateMatFromLog()
    model = SBM(Y, k, clusterAssign = clusterAssign)
    model.variational_inference()

    print(model.B)
    print(np.around(model.gamma, 2))

    gamma = model.gamma
    gamma_mask = get_gamma_mask1(gamma)
    print(gamma_mask)
    order, error = util.alignClusterAssignAndGamma(clusterAssign, gamma_mask)


    print(clusterAssign)
    print(gamma[:, order])
    print(model.B)
    concated = np.concatenate((clusterAssign, gamma[:, order]), axis = 1)
    print(concated)
    

if __name__ == '__main__':
    main()