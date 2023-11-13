from musicflower.webapp import WebApp, heatmap_visualiser, fourier_visualiser

if __name__ == '__main__':

    app = WebApp(verbose=True)  # print additional info registered features, callbacks etc.
    app.use_chroma_features(2000)  # max resolution of 2000
    app.use_fourier_features()
    app.register_visualiser('Chroma Features', 'chroma-features', heatmap_visualiser)
    app.register_visualiser('Fourier Coefficients', 'fourier-features', fourier_visualiser)
    app.init(suppress_flask_logger=True)  # suppress extensive logging, only show errors
    app.run()
