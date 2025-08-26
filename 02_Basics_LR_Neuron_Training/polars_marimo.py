import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from polars import selectors as cs
    return (pl,)


@app.cell
def _():
    import os
    print(os.getcwd())

    return


@app.cell
def _(pl):
    df = pl.read_csv("./02_Pytorch_examples/used_cars.csv", separator=",")
    return (df,)


@app.cell
def _(df):
    print(len(df))
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df["model"], df["model_year"]
    return


@app.cell
def _(df):
    df["model_year"].plot.hist()
    return


@app.cell
def _(df):
    df["model_year"].mean(), df["model_year"].min(), df["model_year"].median(),
    return


@app.cell
def _(df, pl):
    df_1 = (
        df
        .with_columns(
            pl.col('price')
            .str.replace('\\$', '')
            .str.replace(',', '')
            .str.replace(',', '')
            .cast(pl.Int64))
    )
    return (df_1,)


@app.cell
def _(df_1, pl):
    df_2 = (
        df_1
        .with_columns(
            pl.col('mileage')
            .str.replace('mi', '')
            .str.replace(',', '')
            .str.replace(',', '')
            .cast(pl.Int64))
    )
    
    return (df_2,)


@app.cell
def _(df_2):
    df_2
    return


@app.cell
def _(df_1, pl):
    df_1.filter(pl.col('price') < 300000)['price'].plot.hist()
    return


@app.cell
def _(df_1, pl):
    df_1.filter(pl.col("brand").eq("Ford"))["price"].mean()
    return


@app.cell
def _(df_1):
    df_1["brand"].to_dummies()
    # in pandas it is get_dummies
    #we could use the one hot encoded brand columns as predictors but we would ave quite an amount of columns for only 4000 data points
    return


@app.cell
def _(df_1):
    df_1.select("model_year", "price").corr()
    #setting identical values in both columns creates a correlation of 1
    return


@app.cell
def _(df_1):
    df_1.describe()
    return


app._unparsable_cell(
    r"""
    df_1.select(\"model_year\", \"price\").plot.scatter(x=\"model_year\", y=\"price\").

    """,
    name="_"
)


@app.cell
def _(df):
    df_model = df.select()
    return


if __name__ == "__main__":
    app.run()
