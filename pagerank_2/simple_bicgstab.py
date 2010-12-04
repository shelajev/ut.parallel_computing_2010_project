class Solver:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = np.zeros((len(A),1.0))
        #self.x = np.matrix([[4.22578],[4.22859],[7.93882]])
        self.result = None
        self.convergence = 0.0001
    
    def mex(self, matrix, vector):
        return matrix * vector
    
    def dot(self, a, b):
        return np.tensordot(a, b)
    
    def bicgstab(self, iterations = 10000):
        # provide local variables
        mex = self.mex
        dot = self.dot
        A = self.A
        x = self.x
        b = self.b
        convergence = self.convergence
        
        r = b - mex(A, x)
        r_hat = r.copy()
        rho = alpha = w = 1.0
        v = np.zeros((len(A),1.0))
        p = v.copy()
        self.init = True
        
        # iterations
        for i in range(iterations):
            rho_i = dot(r_hat, r)
            beta  = (rho_i / rho) * (alpha / w)
            p_i   = r + beta * (p - w * v)
            v_i   = mex(A, p_i)
            alpha = rho_i / dot(r_hat, v_i)
            s     = r - alpha * v_i
            t     = mex(A, s)

            w_i   = dot(t, s) / dot(t, t)
            x_i   = x + alpha * p_i + w_i * s
            
            r_i   = s - w_i * t
            
            if abs((x_i - x).sum()) < convergence:
                break
            
            # shift
            r   = r_i
            rho = rho_i
            v = v_i
            p = p_i
            w = w_i
            x = x_i
        
        self.x = x
        self.r = r

def test():
    A = np.matrix([[1,2,3],[0,4,5],[0,0,1]])
    b = np.matrix([[5],[3],[7]])
    s = Solver(A, b)
    s.bicgstab()
    print "---------"
    print "--done---"
    print "---------"
    print "-x-------"
    print s.x
    print "-b-------"
    print b
    print "-Ax------"
    print s.A * s.x
test()