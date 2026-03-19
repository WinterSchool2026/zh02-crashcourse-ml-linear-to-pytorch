import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import tqdm

if __name__ == "__main__":
    #===========================================================================
    # Chose your favorite Site number between 0 and 99
    data = pd.read_csv("../data/sites_0.csv")
    #===========================================================================

    #load the data
    time = np.array(data["time"])
    year = np.array(data["year"])
    t2m = np.array(data["t2m"])
    tp = np.array(data["tp"])
    ssrd = np.array(data["ssrd"])
    swvl1 = np.array(data["swvl1"])
    swvl2 = np.array(data["swvl2"])
    swvl3 = np.array(data["swvl3"])
    swvl4 = np.array(data["swvl4"])

    #make numpy arrays out of it
    x = np.stack([t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, time], axis=-1)
    y = np.array(data["lai"])

    #Normalize the features
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)

    #Shuffle the examples to split into train and test
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    x_train = x[: 26 * 24]
    x_test = x[26 * 24 :]
    y_train = y[: 26 * 24]
    y_test = y[26 * 24 :]
    
    #Plot the data
    plt.figure(figsize=(6, 6))
    plt.scatter(x_train[:, 0], y_train, color="#005555", label="training data")
    plt.scatter(x_test[:, 0], y_test, color="#ef7c00", label="test data")
    plt.xlabel("Temperature [K]")
    plt.ylabel("LAI [-]")
    plt.legend()
    plt.show()
    plt.close()

    #===========================================================================
    # Here we want to find what values on k lead to over/underfitting. Therefore
    # make a list of different ks to test.
    ks = [1, 624]
    #===========================================================================
    mse_train = []
    mse_test = []

    # Run the algorithm for each k
    for k in ks:
        # Fit a knn to the data
        knn = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train)
        # make predictions on the training set
        predictions = knn.predict(x_train)
        # calculate the mean squarred error on the training set
        mse_train.append(np.mean((predictions - y_train) ** 2))

        # make predictions on the test set
        predictions = knn.predict(x_test)
        # calculate the mean squarred error on the test set
        mse_test.append(np.mean((predictions - y_test) ** 2))

    # plot the relation between k and the errors on train and test set
    plt.figure()
    plt.plot(ks, mse_train, color = '#007777', lw = 3, label = 'training error')
    plt.plot(ks, np.array(mse_test) - np.array(mse_train), lw = 3, color = '#ef7c00', label = 'generalization error')
    plt.legend()
    plt.show()
    plt.close()


    #===========================================================================
    # Experiment 2: Generalize across locations
    #===========================================================================
    x_train = []
    y_train = []

    # load the first 80 sites as training data
    for nr in range(80):
        data = pd.read_csv(f"../data/sites_{nr}.csv")

        time = np.array(data["time"])
        year = np.array(data["year"])
        t2m = np.array(data["t2m"])
        tp = np.array(data["tp"])
        ssrd = np.array(data["ssrd"])
        swvl1 = np.array(data["swvl1"])
        swvl2 = np.array(data["swvl2"])
        swvl3 = np.array(data["swvl3"])
        swvl4 = np.array(data["swvl4"])
        lat = np.array(data["lat"])
        lon = np.array(data["lon"])

        x = np.stack([t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, year, time, lat, lon], axis=-1)
        y = np.array(data["lai"])

        x_train.append(x)
        y_train.append(y)

    # make it into an array
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # load the last twventy sites as test data
    x_test = []
    y_test = []

    for nr in range(80, 100):
        data = pd.read_csv(f"../data/sites_{nr}.csv")

        time = np.array(data["time"])
        year = np.array(data["year"])
        t2m = np.array(data["t2m"])
        tp = np.array(data["tp"])
        ssrd = np.array(data["ssrd"])
        swvl1 = np.array(data["swvl1"])
        swvl2 = np.array(data["swvl2"])
        swvl3 = np.array(data["swvl3"])
        swvl4 = np.array(data["swvl4"])
        lat = np.array(data["lat"])
        lon = np.array(data["lon"])

        x = np.stack([t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, year, time, lat, lon], axis=-1)
        y = np.array(data["lai"])

        x_test.append(x)
        y_test.append(y)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    # plot the data
    plt.figure(figsize=(6, 6))
    plt.scatter(x_train[:, 0], y_train, color="#005555", label="training data")
    plt.scatter(x_test[:, 0], y_test, color="#ef7c00", label="test data")
    plt.xlabel("Temperature [K]")
    plt.ylabel("LAI [-]")
    plt.legend()
    plt.show()
    plt.close()

    #===========================================================================
    # Here again choose some good values for k:
    ks = [1]
    #===========================================================================
    mse_train = []
    mse_test = []

    for k in tqdm.tqdm(ks):
        #same as above but with more data
        knn = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train)
        predictions = knn.predict(x_train)
        mse_train.append(np.mean((predictions - y_train) ** 2))

        predictions = knn.predict(x_test)
        mse_test.append(np.mean((predictions - y_test) ** 2))
    
    #plot the same as above
    plt.figure()
    plt.plot(ks, mse_train, color = '#007777', lw = 3, label = 'training error')
    plt.plot(ks, np.array(mse_test) - np.array(mse_train), lw = 3, color = '#ef7c00', label = 'generalization error')
    plt.legend()
    plt.show()
    plt.close()

    #===========================================================================
    # Task 3: Upscaling to Europe
    #===========================================================================
    
    # Load the data
    data = pd.read_csv("../data/upscaling.csv")

    time = np.array(data["time"])
    year = np.array(data["year"])
    t2m = np.array(data["t2m"])
    tp = np.array(data["tp"])
    ssrd = np.array(data["ssrd"])
    swvl1 = np.array(data["swvl1"])
    swvl2 = np.array(data["swvl2"])
    swvl3 = np.array(data["swvl3"])
    swvl4 = np.array(data["swvl4"])
    lat = np.array(data["lat"])
    lon = np.array(data["lon"])
    coord_x = np.array(data["x"])
    coord_y = np.array(data["y"])

    # Take the coordinates to find the correct locations after predicting
    coords = np.stack([time // 15, coord_x, coord_y], axis=-1)
    upscale_x = np.stack(
        [t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, year, time, lat, lon], axis=-1,
    )
    upscale_y = np.array(data["lai"])

    #===========================================================================
    # Here the idea is first to select a good value for the knn as evaluated
    # above to use for upscaling.
    # 
    # Afterward:
    # Create a linear regression for comparison. Here you can try some stuff
    # if you want.
    linreg = LinearRegression().fit(x_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=1).fit(x_train, y_train)
    #===========================================================================
    
    # make predictions with the knn and the linear regression
    preds = knn.predict(upscale_x)
    preds_lin = linreg.predict(upscale_x)

    # use the coordinates to put the predictions at the correct locations
    output = np.ones((24, 146, 837 - 678)) * float("NaN")
    output_true = np.ones((24, 146, 837 - 678)) * float("NaN")
    output_lin = np.ones((24, 146, 837 - 678)) * float("NaN")
    for nr in tqdm.tqdm(range(coords.shape[0])):
        output[*coords[nr]] = preds[nr]
        output_lin[*coords[nr]] = preds_lin[nr]
        output_true[*coords[nr]] = upscale_y[nr]

    # plot a map of the mean over the year for the knn
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(output.mean(0), cmap="Greens")
    plt.subplot(1, 3, 2)
    plt.imshow(output_true.mean(0), cmap="Greens")
    plt.subplot(1, 3, 3)
    plt.imshow(output_true.mean(0) - output.mean(0), cmap="coolwarm")
    plt.show()
    plt.close()

    # plot a map for the knn and the linear model to compare
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(output.mean(0), cmap="Greens")
    plt.subplot(1, 2, 2)
    plt.imshow(output_lin.mean(0), cmap="Greens")
    plt.show()
    plt.close()

    # calculate the R2 and the MSE for both individual and their mean
    print('Tree R2:', 1 - np.var(preds - upscale_y) / np.var(upscale_y))
    print('Tree MSE:', np.mean((preds - upscale_y)**2))

    print('linear regression R2:', 1 - np.var(preds_lin - upscale_y) / np.var(upscale_y))
    print('linear regression MSE:', np.mean((preds_lin - upscale_y)**2))

    print('sum R2:', 1 - np.var(0.5*(preds_lin + preds) - upscale_y) / np.var(upscale_y))
    print('sum MSE:', np.mean((0.5*(preds_lin + preds) - upscale_y)**2))

