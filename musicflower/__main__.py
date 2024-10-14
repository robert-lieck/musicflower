from musicflower.webapp import WebApp, fourier_visualiser, keyscape, tonnetz_visualiser, spectral_dome

if __name__ == '__main__':

    app = WebApp(verbose=True)  # print additional info registered features, callbacks etc.
    app.use_chroma_features(200)  # max resolution of 200
    app.use_fourier_features()
    app.use_chroma_scape_features()
    app.use_fourier_scape_features()
    app.register_visualiser('Fourier Coefficients',
                            ['fourier-features'],
                            fourier_visualiser)
    app.register_visualiser('Keyscape',
                            ['fourier-scape-features',
                             'chroma-scape-features'],
                            keyscape)
    app.register_visualiser('Tonnetz',
                            ['chroma-features'],
                            tonnetz_visualiser)
    app.register_visualiser('Spectral Dome',
                            ['fourier-scape-features',
                             'chroma-scape-features'],
                            spectral_dome)
    app.init(
        figure_width=1500,
        figure_height=800,
    )
    app.run()
