from numpy import cos, sin, eye, arange, vstack, hstack, zeros, array
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
figWidth = 7.5
figsize = (figWidth, figWidth * 6.0 / 10.0)


def do_hw(N, save=False):
    
    # Set up figures.
    if save:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    
        fig2 = plt.figure(figsize=figsize)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_xlim((0,1))
        #ax2.set_ylim((0,1))
        fig2.suptitle('linear basis functions')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$u$')
    
    h = (1.0 - 0) / (N - 1)
    # Define the left and right pieces of the basis functions as separate linear functions.
    # These are functions that return functions.
    
    af = lambda i: (i - 1) * h    
    
    def  phi_l(i):
        a = af(i)
        return lambda x: (x - a) / h

    def phi_r(i):
        a = af(i)
        b = a + h
        c = b + h
        return lambda x: -(x - c) / h
    
    def integrate(f, a, b, precision=1000):
        '''Simple Reimann-sum integrator. Could be improved by using
        trapezoidal integration. Probably doesn't perform well for high-slope
        functions (f).'''
        dx = (float(b) - float(a)) / float(precision)
        sum_ = 0
        for x in arange(a, b, dx):
            sum_ += f(x) * dx
        return sum_   
        
    # Set up problem. 
    epsilon = 0.5
    omega = 10.0
    m = 4.0
    amp = 18.0
    forcing = lambda x: -amp * sin(x * omega) - m * x
    
    analytical = lambda x: 1 + epsilon*x - (m*x)/2 + (m*x**3)/6 + \
        (amp*x*cos(omega))/omega - amp*sin(omega*x)/omega**2
    
    # Construct the coeffecient matrix.    
    
    main = 2 / h * eye(N)
    tri = -1 / h * eye(N-1)
    tril = vstack((zeros((1, N)), hstack((tri, zeros((N-1, 1)) )) ))
    triu = vstack((zeros((1, N)), hstack((tri, zeros((N-1, 1)) )) )).T
    A = main + tril + triu
    
    # Include boundary conditions' modifications.
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = A[-1, -1] * 0.5
    
    # Construct the right-hand-side column vector.
    rhs = []
    for i in range(N):
        a = af(i)  # x=a at the left edge of the ith basis function, 
        b = a + h  # b at the point,
        c = b + h  # & c at the right edge.
        
        # Create functions to be integrated for the right-hand-side vector.
        f_l = lambda x: forcing(x) * phi_l(i)(x)
        f_r = lambda x: forcing(x) * phi_r(i)(x)
        if i==N-1:  # right (Neumann) boundary condition
            rhs.append(integrate(f_l, a, b) + epsilon)
        else:
            rhs.append(integrate(f_l, a, b) + integrate(f_r, b, c))
    
    rhs = array(rhs).reshape((N, 1))
    rhs[0] = 1  # left (Dirichlet) boundary condition
    
    # Solve for the coeffecients of the basis functions.
    # Another option here is github.com/tsbertalan/openmg
    u = solve(A, rhs)
    
    # Plot scaled basis functions.
    for i in range(N):
        a = af(i)
        b = a + h
        c = b + h    
        phi_i_l = phi_l(i)
        xl_l = list(arange(a, b, (b - a) / 100))
        yl_l = [phi_i_l(x) * u[i] for x in xl_l]
        
        phi_i_r = phi_r(i)
        xl_r = list(arange(b, c, (c - b) / 100))
        yl_r = [phi_i_r(x) * u[i] for x in xl_r]
        
        if save:
            if i==N-1:  # We only want one label in the plot legend, not N labels.
                ax2.plot(xl_l, yl_l, 'k--', label='scaled linear basis functions, left sides')
                ax2.plot(xl_r, yl_r, 'k-', label='scaled linear basis functions, right sides')
            else:
                ax2.plot(xl_l, yl_l, 'k--')
                ax2.plot(xl_r, yl_r, 'k-')
    
    # Compose the finite element solution as the sum of the product of
    # coeffecients u[j] and basis functions phi_l(i) and phi_r(i).
    FE_domains = []
    FE_soln = []
    for i in range(1, N):
        # in each node, two basis functions apply.
        phi_lower = phi_r(i-1)
        phi_upper = phi_l(i)
        node_soln_function = lambda x: u[i-1] * phi_lower(x) + u[i] * phi_upper(x)
        # The node streches from a to b, a distance of length h.
        a = af(i)
        b = a + h
        node_domain = list(arange(a, b, (b - a) / 100))
        node_soln = [node_soln_function(x) for x in node_domain] 
        # Add these to the growing solution array 
        FE_soln.extend(node_soln)
        FE_domains.extend(node_domain)
    if save:
        xl = list(arange(0, 1, h))
        xl_fine = list(arange(0, 1, h / 10.0))  # For plotting the analytical solution.
        if len(xl) < len(u):  # For very small N (~4), we sometimes lose x=1.0
            xl.append(1.0)
        ax2.plot(FE_domains, FE_soln, 'r-', label='FE solution')
        ax.plot(FE_domains, FE_soln, 'r-', label='FE solution')

        # ax2.plot(xl_fine, [analytical(x) for x in xl_fine], 'r-', label='analytical solution')
        # ax.scatter(xl, u, label=r'$N=%i$ finite element coeffecients' % N, color='k', marker='d')
        # ax.plot(xl_fine, [-forcing(x) for x in xl_fine], 'k--', label=r'forcing function $-f(x)=%.2f sin(%.1f x)+%.1f x$' % (amp, omega, m))
    
        ax.plot(xl_fine, [analytical(x) for x in xl_fine], 'k--', label="analytical (via Green's Function")
        fig.suptitle(r'Solution of $\frac{\partial^2 u(x) }{ \partial x^2 }= -f(x)$, BC: $u(0)=1$, $u\'(1)=%.1f$, %i Finite Elements' % (epsilon, N))
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
        ax.set_xlim((0, 1))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
    
        fig.savefig('hw7-solution_and_forcing-N%i.pdf' % N)
        fig2.savefig('hw7-basis_functions-N%i.pdf' % N)
    
    # The error is the vector difference of the finite element solution and
    # the Green's function solution, at the same x points.
    error = []
    for (FE, a) in zip(FE_soln, [analytical(x) for x in FE_domains]):
        error.append(abs(FE - a))
    return error

if __name__=="__main__":
    # For each of several choices of norm, plot the decreasing norm of
    # the error as the number of basis functions increases.
    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(1, 1, 1)
    to_save = [5, 20, 100]
    N_list = []
    Nmin = 2
    Nmax = 101
#    Nmax = 20
    orders=[0, 1, -1, 2, -2]
    norm_list = [[] for order in orders]
    ordernames=['order of %i' % order for order in orders]
    for N in range(Nmin, Nmax):
        print 'N is', N
        if N in to_save:
            error = do_hw(N, save=True)
        else:
            error = do_hw(N, save=False)
        N_list.append(N)
        for normIndex in range(len(orders)):
            norm_list[normIndex].append(norm(error, ord=order))
    ax3.set_yscale('log')
    fig3.suptitle('2-norm of the error')
    ax3.set_xlim((Nmin, Nmax))
    ax3.set_xlabel('Number of Basis Functions')
    ax3.set_ylabel('norm(FE_solution - Analytical_solution)')
    for normIndex in range(len(orders)):
        ax3.plot(N_list, norm_list[normIndex], label=ordernames[normIndex])
    ax3.legend()
    fig3.savefig('hw7-error_rate-mult_orders.pdf')
