import pandas as pd
from scipy import stats
import numpy as np
from sklearn.metrics import roc_curve, auc

COLUMNS_NAME = ['COD_PROGRAMA',
                'JORNADA',
                'DURACION',
                'PERIODO_INGRESO',
                #'SEXO',
                'ESTADO_CIVIL',
                'ESTRATO',
                'RANGO_EDAD',
                #'CONDICION_EXCEPCION',
                'TIPO_COLEGIO',
                'RANGO_INGRESOS',
                'RANGO_GASTOS',
                'TIPO_VIVIENDA',
                'PUNTAJE_ICFES',
                'PCN',
                'PLC',
                'PMA',
                'PSC',
                'PIN']

def cliff_delta(X, Y):
    """Calculate the effect size using the Cliff's delta."""
    lx = len(X)
    ly = len(Y)
    mat = np.zeros((lx, ly))
    for i in range(0, lx):
        for j in range(0, ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    return (np.sum(mat)) / (lx * ly)


def compute_features_deviations(diff_df, test_df, name_id_label, name_var_target, normal_label, abnormal_label):
    """ Calculate the Cliff's delta effect size between groups."""
    features_df = pd.DataFrame(columns=['features', 'pvalue', 'effect_size'])

    normal_df = test_df[test_df[name_var_target]==normal_label][name_id_label]
    abnormal_df = test_df[test_df[name_var_target]==abnormal_label][name_id_label]
    
    diff_normal = diff_df.loc[normal_df]
    diff_abnormal = diff_df.loc[abnormal_df]

    for feature in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_normal[feature], diff_abnormal[feature])
        effect_size = cliff_delta(diff_abnormal[feature].values, diff_normal[feature].values)

        features_df = features_df.append({'features': feature, 'pvalue': pvalue, 'effect_size': effect_size},
                                     ignore_index=True)

    return features_df

def compute_classification_performance(reconstruction_error_df, test_df, name_id_label, name_var_target, normal_label, abnormal_label):
    """ Calculate the AUCs of the normative model."""
    normal_df = test_df[test_df[name_var_target]==normal_label][name_id_label]
    abnormal_df = test_df[test_df[name_var_target]==abnormal_label][name_id_label]
    
    error_normal = reconstruction_error_df.loc[normal_df]['Reconstruction error']
    error_abnormal = reconstruction_error_df.loc[abnormal_df]['Reconstruction error']
    

    fpr, tpr, _ = roc_curve(list(np.zeros_like(error_normal)) + list(np.ones_like(error_abnormal)),
                            list(error_normal) + list(error_abnormal))

    roc_auc = auc(fpr, tpr)

    tpr = np.interp(np.linspace(0, 1, len(COLUMNS_NAME)), fpr, tpr)

    tpr[0] = 0.0

    return roc_auc, tpr