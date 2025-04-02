from tabdd.models.utils import pretty_encoder_name

DATASETS = {
    'Adult': 'Adult',
    'AmazonEmployeeAccess': 'Amazon Employee Access',
    'BankMarketing': 'Bank Marketing',
    'Higgs': 'Higgs',
    'MedicalAppointments': 'MedicalAppointments',
    'NumerAI': 'Numer AI',
    'PhishingWebsites': 'Phishing Websites',
    'TencentCTRSmall': 'Tencent CTR',
}

AUTOENCODERS = [
    'MLPAutoEncoder', 'GNNAutoEncoder',
]

MULTIHEAD_AUTOENCODERS = [
    'MultiHeadMLPAutoEncoder', 'MultiHeadGNNAutoEncoder',
]

# DISTILL_METHODS = {
#    'random_sample': 'Random Sample',
#    'kmeans': 'KMeans',
#    'agglo': 'Agglo',
#    'kip': 'KIP',
# }

DISTILL_SPACES = [
   'encoded',
   'original',
]

OUTPUT_SPACES = [
   'encoded',
   'decoded',
]

CLUSTER_CENTERS = [
   'closest',
   'centroid',
]

DISTILL_METHODS = [
   'agglo',
   'kmeans',
   'kip',
]

DATA_MODES = [
    'Original',
    'Mixed Original',
    'Random Sample', 
    'KMeans -> Original / Centroid',
    'KMeans -> Original / Closest',
    'Agglo -> Original / Centroid',
    'Agglo -> Original / Closest',
    'KIP -> Original / Centroid',
]

for enc in AUTOENCODERS + MULTIHEAD_AUTOENCODERS:
   DATA_MODES += [
        f'{pretty_encoder_name(enc)} Encoded', 
        f'{pretty_encoder_name(enc)} Decoded', 
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Encoded / Centroid',
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Encoded / Closest',
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Decoded / Centroid',
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Decoded / Closest',
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Decoded-Binary / Centroid',
        f'KMeans-{pretty_encoder_name(enc)}-Encoded -> Decoded-Binary / Closest',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Encoded / Centroid',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Encoded / Closest',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Decoded / Centroid',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Decoded / Closest',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Decoded-Binary / Centroid',
        f'Agglo-{pretty_encoder_name(enc)}-Encoded -> Decoded-Binary / Closest',
        f'KIP-{pretty_encoder_name(enc)}-Encoded -> Encoded / Centroid',
        f'KIP-{pretty_encoder_name(enc)}-Encoded -> Decoded / Centroid',
        f'KIP-{pretty_encoder_name(enc)}-Encoded -> Decoded-Binary / Centroid',
   ]

DISTILL_DATA_MODES = [
    'Random Sample',
    'KMeans Original',
    'Closest KMeans Original',
    'Agglo Original',
    'Closest Agglo Original',
    'KIP Original',
    'KMeans Encoded',
    'Closest KMeans Encoded',
    'Agglo Encoded',
    'Closest Agglo Encoded',
    'KIP Encoded',
]

DATASET_SCALE_MODES = {
    'none': 'None',
    'standard': 'Standard',
}

CLASSIFIERS = [
    'XGBClassifier', 
    'LogisticRegression', 
    'MLPClassifier', 
    'KNeighborsClassifier',
    'GaussianNB'
]

CLASSIFIER_METRICS = {
    'balanced_accuracy': 'Balanced Accuracy',
    'f1_weighted': 'Weighted F1',
}

REDUCE_METHODS = [
    'PCA',
    'UMAP',
    'TSNE',
]

DISTILL_SIZES = [
    ((i+1)*10) for i in range(10)
]

PLOT_DOWNLOAD_CONFIG = {
        'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'plot',
        'scale':4 # Multiply title/legend/axis/canvas sizes by this factor
    }
}
