About Dataset
Context

Forest fires can be devastating to ecosystems, wildlife, and human infrastructure. Predicting the burned area based on environmental variables can aid in taking preventative efforts to reduce the impact of forest fires. This dataset comprises several characteristics that influence forest fires in Montesinho Park, Portugal, and is intended to estimate the burned area caused by such fires.
Content

This dataset consists of 12 attributes that capture different aspects of the environmental and spatial conditions affecting forest fires:

    area: the burned area of the forest (in ha), transformed with log(x + 1) to handle zero values gracefully.
    X: x-axis spatial coordinate within the Montesinho park map (1 to 9)
    Y: y-axis spatial coordinate within the Montesinho park map (2 to 9)
    month: month of the year (encoded as integers from 0 to 11 using LabelEncoder)
    day: day of the week (encoded as integers from 0 to 6 using LabelEncoder)
    DMC: Drought Code index from the FWI system (1.1 to 291.3)
    DC: Drought Code index from the FWI system (7.9 to 860.6)
    ISI: Initial Spread Index from the FWI system (0.0 to 56.10)
    temp: temperature in Celsius degrees (2.2 to 33.30)
    RH: relative humidity in % (15.0 to 100)

Acknowledgements

This dataset is publicly available for research purposes. Details are provided in [Cortez and Morais, 2007]. Please include this citation if you want to use this database:

P. Cortez & A. Morais. A Data Mining Approach for Predicting Forest Fires Using Meteorological Data. In J. Neves, M. F. Santos, and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: http://www.dsi.uminho.pt/~pcortez/fires.pdf
Inspiration

Understanding and anticipating the area affected by forest fires can greatly benefit fire management and prevention strategies. By using this dataset, you can explore the relationship between different environmental factors and develop models that help in early warning systems and resource allocation for fire control efforts. 

https://www.kaggle.com/datasets/anitarostami/montesinho-forest-fire-prediction-dataset?resource=download