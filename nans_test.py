import unittest
import pandas as pd
import numpy as np
from nans import NanHandler  

class TestNanHandler(unittest.TestCase):

    def test_handle_infinite_values(self):
        # DataFrame con valores infinitos
        df = pd.DataFrame({
            'A': [1, 2, np.inf],
            'B': [-np.inf, 5, 6],
            'C': [7, 8, 9]
        })

        nan_handler = NanHandler(df)
        nan_handler.handle_infinite_values()

        # Comprobar que los infinitos han sido reemplazados por NaN
        self.assertTrue(np.isnan(nan_handler.df.loc[2, 'A']))
        self.assertTrue(np.isnan(nan_handler.df.loc[0, 'B']))

        # Comprobar que los demás valores se mantienen
        self.assertEqual(nan_handler.df.loc[0, 'A'], 1)
        self.assertEqual(nan_handler.df.loc[1, 'B'], 5)

    def test_detect_nans(self):
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [np.nan, np.nan, 6],
            'C': [7, 8, 9]
        })

        nan_handler = NanHandler(df)
        nan_percentage = nan_handler.detect_nans()

        expected_nan_percentage = pd.Series({
            'A': 33.33333333333333,
            'B': 66.66666666666666,
            'C': 0.0
        })

        pd.testing.assert_series_equal(nan_percentage, expected_nan_percentage)

    def test_get_nan_summary(self):
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [np.nan, np.nan, 6],
            'C': [7, 8, 9]
        })

        nan_handler = NanHandler(df)
        summary = nan_handler.get_nan_summary()

        expected_summary = pd.DataFrame({
            'num_nans': [1, 2, 0],
            'perc_nans': [33.33333333333333, 66.66666666666666, 0.0]
        }, index=['A', 'B', 'C'])

        pd.testing.assert_frame_equal(summary, expected_summary)

    def test_handle_nans(self):
        df = pd.DataFrame({
            'num_col': [1, np.nan, 3],
            'cat_col': ['a', np.nan, 'b'],
            'date_col': [pd.Timestamp('2020-01-01'), pd.NaT, pd.Timestamp('2020-01-03')],
            'high_nan_col': [np.nan, np.nan, np.nan]
        })

        nan_handler = NanHandler(df)

        decisions = {
            'num_col': 'mean',
            'cat_col': 'mode',
            'date_col': 'median',
            'high_nan_col': 'drop_column',
            'drop_rows_with_nans': 'no'
        }

        default_custom_values = {
            'cat_col': 'Unknown',
            'date_col': pd.Timestamp('2020-01-02')
        }

        df_cleaned, columns_needing_decisions = nan_handler.handle_nans(
            decisions=decisions,
            default_custom_values=default_custom_values
        )

        # Verificar que 'high_nan_col' ha sido eliminada
        self.assertNotIn('high_nan_col', df_cleaned.columns)

        # Verificar que 'num_col' ha sido imputada con la media
        expected_num_col_mean = np.mean([1, 3])
        self.assertEqual(df_cleaned.loc[1, 'num_col'], expected_num_col_mean)

        # Verificar que 'cat_col' ha sido imputada con la moda o valor predeterminado
        self.assertEqual(df_cleaned.loc[1, 'cat_col'], 'a')

        # Verificar que 'date_col' ha sido imputada con la mediana
        expected_median_date = pd.to_datetime('2020-01-02')
        self.assertEqual(df_cleaned.loc[1, 'date_col'], expected_median_date)

        # Verificar que no hay columnas pendientes de decisiones
        self.assertEqual(columns_needing_decisions, [])

    def test_advanced_imputation(self):
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan],
            'C': ['x', 'y', 'z']  # Columna categórica que no debería ser afectada
        })

        nan_handler = NanHandler(df)
        nan_handler.advanced_imputation(method='knn', n_neighbors=2)

        # Verificar que los NaNs han sido imputados
        self.assertFalse(nan_handler.df[['A', 'B']].isna().any().any())

        # Verificar que la columna categórica no ha sido modificada
        pd.testing.assert_series_equal(nan_handler.df['C'], df['C'])

    def test_visualize_nans_bars(self):
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, np.nan, 6],
            'C': [np.nan, np.nan, np.nan]
        })

        nan_handler = NanHandler(df)
        # Probamos que el método se ejecuta sin errores
        try:
            nan_handler.visualize_nans_bars(show_plot=False, save_plot=False)
            result = True
        except Exception as e:
            print(f"Error al ejecutar visualize_nans_bars: {e}")
            result = False

        self.assertTrue(result)

