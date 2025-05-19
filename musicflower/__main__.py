from musicflower.webapp import WebApp
import musicflower.features as f
import musicflower.visualisers as v

if __name__ == '__main__':
    app = WebApp(verbose=True)         # print additional info about registered features, callbacks etc.
    f.use_chroma_features(app, n=200)  # max temporal resolution of 200
    v.use_fourier_visualiser(app)      # register visualiser and required features
    v.use_keyscape_visualiser(app)
    v.use_tonnetz_visualiser(app)
    v.use_spectral_dome_visualiser(app)
    app.init(
        figure_width=1500,
        figure_height=800,
    )
    app.run()
