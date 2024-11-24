# import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import PCA fomr sklearn
from sklearn.decomposition import PCA
# import umap from umap
import umap


# load the data
path = r'G:\My Drive\Opto\A_EYE_algorithm\latent\latent'
df_test = pd.read_csv(path + r'\z_test.csv')

# fit PCA
pca = PCA(n_components=3)
pca_result_test = pca.fit_transform(df_test.values)
df_pca_test = pd.DataFrame(pca_result_test, columns=['pca-one', 'pca-two', 'pca-three'])

# load json file with labels
import json
with open(path + r'\labels_test.json') as f:
    lbl = json.load(f)

# add labels to the dataframe
df_pca_test['lbl'] = lbl

# how much variation did each PC explain?
print(pca.explained_variance_ratio_)

# plot 3d PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_test['pca-one'], df_pca_test['pca-two'], df_pca_test['pca-three'])
plt.show()

# save PCA results to csv
df_pca_test.to_csv(path + r'\pca_test.csv', index=False)
# Save the tensor to a file
np.savetxt(path + r'\pca_test.tsv', df_pca_test.values, delimiter='\t', fmt='%f')


# load other csv files as well, and merge to one dataframe
files = [
    "z_067_all.csv",
    #"z_067_lbl.csv",
    "z_121_all.csv",
    #"z_121_lbl.csv",
    "z_145_all.csv",
    #"z_145_lbl.csv",
]

df = pd.DataFrame()
for file in files:
    df_temp = pd.read_csv(path + r'\\' + file)
    # add the file name as a column
    df_temp['file'] = file
    df = pd.concat([df, df_temp], axis=0)

# transform the data with PCA, using already fitted PCA
pca_result_3slides = pd.DataFrame(pca.transform(df.iloc[:,:-1].values))
pca_result_3slides['file'] = df['file'].values

# plot pca_result_3slides in 3d as well together with pca_test
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
# for each 'file' in the dataframe, plot the data with different color
colors = ['r', 'g', 'b']
for file in df['file'].unique():
    ax.scatter(pca_result_3slides.loc[pca_result_3slides['file'] == file, 0],
                pca_result_3slides.loc[pca_result_3slides['file'] == file, 1],
                pca_result_3slides.loc[pca_result_3slides['file'] == file, 2],
                label=file,
                color=colors.pop(0),
                marker='s',
                alpha=1,
                s=100
                )

# add pca_test to the plot
ax.scatter(
    df_pca_test["pca-one"],
    df_pca_test["pca-two"],
    df_pca_test["pca-three"],
    c="gray",
    marker="o",
    s=50,
    alpha=0.5,
)
plt.show()

# plot 2D PCA
fig, ax = plt.subplots(figsize=(20, 20))
# for each 'file' in the dataframe, plot the data with different color
colors = ['r', 'g', 'b']
for file in df['file'].unique():
    ax.plot(pca_result_3slides.loc[pca_result_3slides['file'] == file, 0],
               pca_result_3slides.loc[pca_result_3slides['file'] == file, 1],
               label=file,
               color=colors.pop(0),
               marker='s',
               alpha=1,               
               linewidth=1)


# use seaborn to plot the data and color by 'lbl' column
sns.scatterplot(
    x=df_pca_test["pca-one"],
    y=df_pca_test["pca-two"],
    hue=df_pca_test["lbl"],
    s=100,
    alpha=0.5,
    ax=ax,
    palette = 'tab10',
    style=df_pca_test["lbl"]

)
ax.set_xlabel('PCA One')
ax.set_ylabel('PCA Two')
ax.legend()
plt.show()


# %% UMAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import umap

# load the data
path = r'G:\My Drive\Opto\A_EYE_algorithm\latent\latent'
df_test = pd.read_csv(path + r'\z_test.csv')

# fit UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result_test = reducer.fit_transform(df_test.values)
#df_umap_test = pd.DataFrame(umap_result_test, columns=['umap-one', 'umap-two', 'umap-three'])
df_umap_test = pd.DataFrame(umap_result_test, columns=['umap-one', 'umap-two'])

# load json file with labels
with open(path + r'\labels_test.json') as f:
    lbl = json.load(f)

# add labels to the dataframe
df_umap_test['lbl'] = lbl

# plot 3d UMAP
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_umap_test['umap-one'], df_umap_test['umap-two'], df_umap_test['umap-three'])
plt.show()

# save UMAP results to csv
df_umap_test.to_csv(path + r'\umap_test.csv', index=False)
# Save the tensor to a file
np.savetxt(path + r'\umap_test.tsv', df_umap_test.values, delimiter='\t', fmt='%f')

# load other csv files as well, and merge to one dataframe
files = [
    #"z_067_all.csv",
    "z_067_lbl.csv",
    #"z_121_all.csv",
    "z_121_lbl.csv",
    #"z_145_all.csv",
    "z_145_lbl.csv",
]

df = pd.DataFrame()
for file in files:
    df_temp = pd.read_csv(path + r'\\' + file)
    # add the file name as a column
    df_temp['file'] = file
    df = pd.concat([df, df_temp], axis=0)

# transform the data with UMAP
umap_result_3slides = pd.DataFrame(reducer.transform(df.iloc[:,:-1].values))
umap_result_3slides['file'] = df['file'].values

# plot umap_result_3slides in 3d together with umap_test
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
# for each 'file' in the dataframe, plot the data with different color
colors = ['r', 'g', 'b']
for file in df['file'].unique():
    ax.scatter(umap_result_3slides.loc[umap_result_3slides['file'] == file, 0],
                umap_result_3slides.loc[umap_result_3slides['file'] == file, 1],
                umap_result_3slides.loc[umap_result_3slides['file'] == file, 2],
                label=file,
                color=colors.pop(0),
                marker='s',
                alpha=1,
                s=100
                )

# add umap_test to the plot
ax.scatter(
    df_umap_test["umap-one"],
    df_umap_test["umap-two"],
    df_umap_test["umap-three"],
    c="gray",
    marker="o",
    s=50,
    alpha=0.5,
)
plt.show()

# plot 2D UMAP
fig, ax = plt.subplots(figsize=(20, 20))
# for each 'file' in the dataframe, plot the data with different color
colors = ['r', 'g', 'b']
for file in df['file'].unique():
    ax.plot(umap_result_3slides.loc[umap_result_3slides['file'] == file, 0],
               umap_result_3slides.loc[umap_result_3slides['file'] == file, 1],
               label=file,
               color=colors.pop(0),
               marker='s',
               alpha=1,               
               linewidth=1)

# use seaborn to plot the data and color by 'lbl' column
sns.scatterplot(
    x=df_umap_test["umap-one"],
    y=df_umap_test["umap-two"],
    hue=df_umap_test["lbl"],
    s=100,
    alpha=0.5,
    ax=ax,
    palette='tab10',
    style=df_umap_test["lbl"]
)
ax.set_xlabel('UMAP One')
ax.set_ylabel('UMAP Two')
ax.legend()
plt.show()

# %%% 

# load other csv files
files = [
    #"z_067_all.csv",
    "z_067_lbl.csv",
    #"z_121_all.csv",
    "z_121_lbl.csv",
    #"z_145_all.csv",
    "z_145_lbl.csv",
]

# Create a separate plot for each file
for file in files:
    # Read the file
    df_temp = pd.read_csv(path + r'\\' + file)
    
    # Transform the data with UMAP
    umap_result = pd.DataFrame(reducer.transform(df_temp.iloc[:,:-1].values))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the test data points
    sns.scatterplot(
        data=df_umap_test,
        x="umap-one",
        y="umap-two",
        hue="lbl",
        s=100,
        alpha=0.5,
        ax=ax,
        palette='tab10',
        style="lbl"
    )
    
    # Plot the file data points
    ax.scatter(
        umap_result[0],
        umap_result[1],
        color='red',
        marker='s',
        s=200,
        alpha=0.7,        
    )
    # Add text index to each point
    for i, txt in enumerate(df_temp.index):
        ax.annotate(txt, (umap_result.iloc[i, 0], umap_result.iloc[i, 1]), fontsize=12, color='black', ha='center', va='center')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'UMAP representation: {file}')
    plt.tight_layout()
    # save the plot
    plt.savefig(path + r'\\' + file[:-4] + '_umap.png')

    plt.show()


# %%%
#  load other csv files
files = [
    "z_067_all.csv",
    #"z_067_lbl.csv",
    "z_121_all.csv",
    #"z_121_lbl.csv",
    "z_145_all.csv",
    #"z_145_lbl.csv",
]

# Create a separate plot for each file
for file in files:
    # Read the file
    df_temp = pd.read_csv(path + r'\\' + file)
    
    # Transform the data with UMAP
    umap_result = pd.DataFrame(reducer.transform(df_temp.iloc[:,:-1].values))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the test data points
    sns.scatterplot(
        data=df_umap_test,
        x="umap-one",
        y="umap-two",
        hue="lbl",
        s=100,
        alpha=0.5,
        ax=ax,
        palette='tab10',
        style="lbl"
    )
    
    # Plot the file data points
    ax.plot(
        umap_result[0],
        umap_result[1],
        color='red',
        marker='.',
        linewidth=1,
        #s=100,
        alpha=0.7,   

    )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'UMAP representation: {file}')
    plt.tight_layout()
    # save the plot
    plt.savefig(path + r'\\' + file[:-4] + '_umap.png')

    plt.show()
# %%
