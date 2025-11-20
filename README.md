# CODE
Correlated latent space learning and continuum modeling of single cell data

adata1.obsm[f'X_umap_{n}'] = adata1.obsm['X_umap']
E_grid, V_grid = ag_code.get_vfres(adata1, zs_key='X_latent', E_key=f'X_umap_{n}')
ax = sc.pl.embedding(adata1, color=f'time_{n}', basis=f'X_umap_{n}', s=17.5, title=f'{n} UMAP projection', cmap='viridis', show=False, frameon=False)
lengths = np.sqrt((V_grid*V_grid).sum(0))
stream_linewidth = 1
stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
stream_kwargs = dict(
    linewidth = stream_linewidth,
    density = 1,
    zorder = 3,
    color = 'k',
    arrowsize = 1,
    arrowstyle = '-|>',
    maxlength = 4,
    integration_direction = 'both',
)
ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
ax.set_xlabel('')
ax.set_ylabel('')
