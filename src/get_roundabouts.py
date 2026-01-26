import osmnx as ox

# On prend une zone plus large que Paris (ex: Ile-de-France ou un département vert)
# Les ronds-points sont rois en banlieue et en province.
place_name = "Yvelines, France" # Ou "Essonne, France" pour du très lourd en ronds-points

# On cherche TOUT ce qui est marqué comme jonction tournante
tags = {'junction': 'roundabout'}

# On récupère les "ways" (les routes) au lieu des "features" pour être plus précis
gdf = ox.features_from_place(place_name, tags=tags)

# Nettoyage : un rond-point est souvent composé de plusieurs petits segments.
# On va grouper par géométrie pour ne pas compter 8 segments pour 1 seul rond-point.
if not gdf.empty:
    # On projette pour calculer le centre correctement
    gdf_projected = gdf.to_crs(epsg=2154)
    
    # On crée un tampon (buffer) et on fusionne ce qui se touche
    unified = gdf_projected.buffer(10).unary_union
    
    # On récupère les centres des polygones fusionnés
    if unified.geom_type == 'MultiPolygon':
        centers = [poly.centroid for poly in unified.geoms]
    else:
        centers = [unified.centroid]

    print(f"Nombre de ronds-points uniques trouvés : {len(centers)}")