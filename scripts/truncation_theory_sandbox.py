import numpy as np
import matplotlib.pyplot as plt


def tests():
    N = 100_000
    sigma = 1.0
    points = np.random.multivariate_normal(mean=np.array([1.0, 0.0]), cov=np.array([[sigma, 0.0], [0.0, sigma]]),
                                           size=N)

    if False:
        # scatter plot
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()

    magnitudes = np.sqrt((points * points).sum(axis=1))

    if False:
        # histogram of magnitudes
        plt.hist(magnitudes, bins=100)
        plt.show()

    cosines = points[:, 0] / magnitudes

    if True:
        # histogram of cosines
        plt.hist(cosines, bins=100)
        plt.show()

    mean = cosines.mean()
    print("Mean cosine value = ", mean)


def avg_cosine_at_dist(*, distance: float, sigma: float, sample_count: int = 100_000) -> float:
    points = np.random.multivariate_normal(mean=np.array([distance, 0.0]), cov=np.array([[sigma, 0.0], [0.0, sigma]]),
                                           size=sample_count)
    magnitudes = np.sqrt((points * points).sum(axis=1))
    cosines = points[:, 0] / magnitudes
    return cosines.mean()


def main():
    N = 100
    sigma_maxes = np.linspace(0.1, 10, 20)

    colors = np.stack([np.linspace(0.0, 1.0, len(sigma_maxes)), np.linspace(1.0, 0.0, len(sigma_maxes)),
                       np.zeros(len(sigma_maxes))])
    colors = [(r.item(), g.item(), b.item()) for r, g, b in zip(colors[0, :], colors[1, :], colors[2, :])]

    distances = np.linspace(0.0, 5, N)
    for j in range(len(sigma_maxes)):
        avg_cosines = np.zeros(len(distances))
        sigmas = np.concat(
            (np.linspace(0.0, sigma_maxes[j], N // 2, endpoint=False), np.linspace(sigma_maxes[j], 0.0, N - (N // 2))))
        for i in range(len(distances)):
            avg_cosines[i] = avg_cosine_at_dist(distance=distances[i].item(), sigma=sigmas[i].item())
        plt.plot(distances, avg_cosines, color=colors[j])
    plt.show()


if __name__ == "__main__":
    main()
