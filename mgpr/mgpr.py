import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from .minimax_tilting_sampler import TruncatedMVN


# ガウスカーネルの関数を定義
def gaussian_kernel(x1, x2, rho2=1.0, gamma2=1.0):
    return rho2 * np.exp(-((x1 - x2) ** 2) / (2 * gamma2))


def d_gaussian_kernel(x1, x2, rho2, gamma2=1.0):
    return rho2 * (x1 - x2) / gamma2 * np.exp(-((x1 - x2) ** 2) / (2 * gamma2))


def d_d_gaussian_kernel(x1, x2, rho2, gamma2=1.0):
    return (
        rho2
        * (1 - (x1 - x2) ** 2 / gamma2)
        / gamma2
        * np.exp(-((x1 - x2) ** 2) / (2 * gamma2))
    )


class mgpr:
    def __init__(self, t, y, x, sigma2=1.0, rho2=1.0, gamma2=1.0, seed=1234):
        """
        t: 観測値の入力
        y: 観測値の出力
        x: 単調制約を入れる点（細かすぎるとエラーを起こしがち）
        sigma2: 観測ノイズの分散（入力された値を初期値としてparameter_optimizeにより最適化可能）
        rho2: ガウスカーネルのパラメータ1（入力された値を初期値としてparameter_optimizeにより最適化可能）
        gamma2: ガウスカーネルのパラメータ2（入力された値を初期値としてparameter_optimizeにより最適化可能）
        """
        self.t = t
        self.y = y
        self.x = x
        self.n = len(y)
        self.c = len(x)
        self.sigma2 = sigma2
        self.rho2 = rho2
        self.gamma2 = gamma2
        np.random.seed(seed)

    # for parameter optimization
    def negative_loglikelihood(self, params):
        try:
            sigma2 = np.exp(params[0])
            rho2 = np.exp(params[1])
            gamma2 = np.exp(params[2])

            K_tt = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    K_tt[i, j] = gaussian_kernel(self.t[i], self.t[j], rho2, gamma2)

            d_K_tx = np.zeros((self.n, self.c))
            for i in range(self.n):
                for j in range(self.c):
                    d_K_tx[i, j] = d_gaussian_kernel(self.t[i], self.x[j], rho2, gamma2)

            d_d_K_xx = np.zeros((self.c, self.c))
            for i in range(self.c):
                for j in range(self.c):
                    d_d_K_xx[i, j] = d_d_gaussian_kernel(
                        self.x[i], self.x[j], rho2, gamma2
                    )

            Sigma_y = K_tt + sigma2 * np.eye(self.n)

            loglikelihood = multivariate_normal(
                mean=np.zeros(self.n), cov=Sigma_y
            ).logpdf(self.y)

            mean_f_prime = d_K_tx.T @ np.linalg.solve(Sigma_y, self.y)
            var_f_prime = d_d_K_xx - d_K_tx.T @ np.linalg.solve(Sigma_y, d_K_tx)
            scaling = np.diag(1 / np.sqrt(np.diag(var_f_prime)))
            mean_f_prime = scaling @ mean_f_prime
            var_f_prime = scaling @ var_f_prime @ scaling
            loglikelihood += multivariate_normal(
                mean=np.zeros(self.c), cov=var_f_prime
            ).logcdf(-mean_f_prime)

            loglikelihood += -multivariate_normal(
                mean=np.zeros(self.c), cov=d_d_K_xx
            ).logcdf(np.zeros(self.c))

            return -loglikelihood
        except:
            return np.inf

    def parameter_optimize(self):
        initial_params = [np.log(self.sigma2), np.log(self.rho2), np.log(self.gamma2)]
        result = minimize(
            self.negative_loglikelihood, initial_params, method="Nelder-Mead"
        )
        self.sigma2 = np.exp(result.x[0])
        self.rho2 = np.exp(result.x[1])
        self.gamma2 = np.exp(result.x[2])
        print("best parameters:")
        print("sigma2:", self.sigma2)
        print("rho2:", self.sigma2)
        print("gamma2:", self.gamma2)
        print("loglikelihood of best parmeters", -result.fun)

    def sampling(self, s, num_samples):
        m = len(s)

        K_tt = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K_tt[i, j] = gaussian_kernel(
                    self.t[i], self.t[j], self.rho2, self.gamma2
                )

        K_st = np.zeros((m, self.n))
        for i in range(m):
            for j in range(self.n):
                K_st[i, j] = gaussian_kernel(s[i], self.t[j], self.rho2, self.gamma2)

        K_ss = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K_ss[i, j] = gaussian_kernel(s[i], s[j], self.rho2, self.gamma2)

        d_K_tx = np.zeros((self.n, self.c))
        for i in range(self.n):
            for j in range(self.c):
                d_K_tx[i, j] = d_gaussian_kernel(
                    self.t[i], self.x[j], self.rho2, self.gamma2
                )

        d_K_sx = np.zeros((m, self.c))
        for i in range(m):
            for j in range(self.c):
                d_K_sx[i, j] = d_gaussian_kernel(
                    s[i], self.x[j], self.rho2, self.gamma2
                )

        d_d_K_xx = np.zeros((self.c, self.c))
        for i in range(self.c):
            for j in range(self.c):
                d_d_K_xx[i, j] = d_d_gaussian_kernel(
                    self.x[i], self.x[j], self.rho2, self.gamma2
                )

        Sigma_y = K_tt + self.sigma2 * np.eye(self.n)

        mean_f_prime = d_K_tx.T @ np.linalg.solve(Sigma_y, self.y)
        var_f_prime = d_d_K_xx - d_K_tx.T @ np.linalg.solve(Sigma_y, d_K_tx)
        scaling = np.diag(1 / np.sqrt(np.diag(var_f_prime)))
        mean_f_prime = scaling @ mean_f_prime
        var_f_prime = scaling @ var_f_prime @ scaling
        f_prime_samples = (
            TruncatedMVN(
                mean_f_prime.copy(),
                var_f_prime.copy(),
                np.zeros(self.c),
                np.inf * np.ones(self.c),
            )
            .sample(num_samples)
            .T
        )

        cov = (d_K_sx - K_st @ np.linalg.solve(Sigma_y, d_K_tx)) @ scaling
        var_post = (
            K_ss
            - K_st @ np.linalg.solve(Sigma_y, K_st.T)
            - cov @ np.linalg.solve(var_f_prime, cov.T)
        )
        U, S, _ = np.linalg.svd(var_post)
        Lambda_sqrt = np.diag(np.sqrt(S))
        fs_samples_mgp = np.zeros((num_samples, m))
        for i in range(num_samples):
            mean_post = K_st @ np.linalg.solve(Sigma_y, self.y) + cov @ np.linalg.solve(
                var_f_prime, f_prime_samples[i, :] - mean_f_prime
            )
            fs_samples_mgp[i, :] = mean_post + U @ Lambda_sqrt @ np.random.randn(m)

        return fs_samples_mgp

        # mu_f_prime = d_K_tx.T @ np.linalg.solve(Sigma_y, self.y)
        # var_f_prime = d_d_K_xx - d_K_tx.T @ np.linalg.solve(Sigma_y, d_K_tx)
        # samples_f_prime = (
        #     TruncatedMVN(
        #         mu_f_prime,
        #         var_f_prime,
        #         np.zeros(self.c),
        #         np.inf * np.ones(self.c),
        #     )
        #     .sample(num_samples)
        #     .T
        # )

        # cov = np.block([d_K_sx, K_st])
        # var = np.block([[d_d_K_xx, d_K_tx.T], [d_K_tx, K_tt]])
        # U, S, _ = np.linalg.svd(K_ss - cov @ np.linalg.solve(var, cov.T))
        # Lambda_sqrt = np.diag(np.sqrt(S))
        # fs_samples_mgp = np.zeros((num_samples, m))
        # for i in range(num_samples):
        #     mean_fs = cov @ np.linalg.solve(
        #         var,
        #         np.hstack(
        #             [
        #                 samples_f_prime[i, :]
        #                 - d_K_tx.T @ np.linalg.solve(Sigma_y, self.y),
        #                 self.y,
        #             ]
        #         ),
        #     )
        #     fs_samples_mgp[i, :] = mean_fs + U @ Lambda_sqrt @ np.random.randn(m)

        # return fs_samples_mgp
