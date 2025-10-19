import sqlite3
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SQLiteHandler:
    def __init__(self, db_path):
        """
        初始化数据库连接
        :param db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.dataframes = {}  # 缓存 DataFrame，key: 表名

    # ---------------- 数据库操作 ----------------
    def list_tables(self):
        """列出数据库中所有表名"""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = [row[0] for row in self.cursor.execute(query).fetchall()]
        return tables

    def get_all_column_names(self, table_name):
        """获取指定表的所有列名"""
        query = f"PRAGMA table_info({table_name});"
        columns = [col[1] for col in self.cursor.execute(query).fetchall()]
        return columns

    # ---------------- 表加载与缓存 ----------------
    def load_table_to_df(self, table_name):
        """
        将指定表加载为 DataFrame 并缓存到内存
        :param table_name: 表名
        :return: pandas DataFrame
        """
        if table_name not in self.dataframes:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', self.conn)
            self.dataframes[table_name] = df
        return self.dataframes[table_name]

    def get_df(self, table_name):
        """返回缓存的 DataFrame，如果未加载则自动加载"""
        return self.load_table_to_df(table_name)

    # ---------------- DataFrame 操作 ----------------
    def filter_by_range(self, df, column_name, min_val=None, max_val=None):
        """
        根据数值区间筛选 DataFrame
        :param df: DataFrame
        :param column_name: 列名
        :param min_val: 最小值（包含）
        :param max_val: 最大值（包含）
        :return: 筛选后的 DataFrame
        """
        col = pd.to_numeric(df[column_name], errors='coerce')
        if min_val is not None:
            df = df[col >= min_val]
        if max_val is not None:
            df = df[col <= max_val]
        return df.reset_index(drop=True)

    def sort_df(self, df, column_name, ascending=False):
        """
        对指定列排序，返回新 DataFrame
        :param df: DataFrame
        :param column_name: 列名
        :param ascending: 是否升序（默认降序）
        :return: 排序后的 DataFrame
        """
        return df.sort_values(by=column_name, ascending=ascending).reset_index(drop=True)

    # ---------------- 导出 ----------------
    def export_to_csv(self, df, file_path):
        """将 DataFrame 导出为 CSV 文件"""
        df.to_csv(file_path, index=False)
        print(f"已导出 CSV 到: {file_path}")

    # ---------------- 工具 ----------------
    def df_row_count(self, df):
        """返回 DataFrame 行数"""
        return df.shape[0]

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
        print("数据库连接已关闭")


class DataAnalyzer:
    def __init__(self, df):
        """
        初始化分析类
        :param df: pandas DataFrame, 必须包含 create_time 列
        """
        self.df = df.copy()
        self._check_columns()
        self.convert_timestamp()
        self.df.to_csv("backup.csv")
        
    def _check_columns(self):
        """检查必要列是否存在"""
        if 'create_time' not in self.df.columns:
            raise ValueError("DataFrame 必须包含 'create_time' 列")
        # 将互动指标默认转成数值型
        for col in ['liked_count', 'comment_count', 'share_count', 'collected_count']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
    # ---------------- 时间处理 ----------------
    def convert_timestamp(self, ts_column='create_time'):
        """
        将 UNIX 时间戳转化为 datetime 格式，并提取时间特征
        """
        self.df["create_time_ts"] = self.df[ts_column]
        self.df[ts_column] = pd.to_datetime(self.df[ts_column], unit='s')
        self.df['date'] = self.df[ts_column].dt.date
        self.df['hour'] = self.df[ts_column].dt.hour
        self.df['weekday'] = self.df[ts_column].dt.weekday  # 0=周一, 6=周日
        self.df['month'] = self.df[ts_column].dt.month
        self.df['minute'] = self.df[ts_column].dt.minute
        # 计算总互动量
        self.df['total_engagement'] = self.df[['liked_count','comment_count','share_count','collected_count']].sum(axis=1)
        return self.df

    # ---------------- 互动分析 ----------------
    def aggregate_by_time(self, time_unit='hour', metrics=['total_engagement']):
        """
        按时间单位统计互动指标均值
        :param time_unit: 'hour', 'weekday', 'date', 'month'
        :param metrics: 要分析的互动指标列表
        :return: 汇总 DataFrame
        """
        if time_unit not in ['hour', 'weekday', 'date', 'month']:
            raise ValueError("time_unit 必须是 'hour','weekday','date','month'")
        agg_df = self.df.groupby(time_unit)[metrics].mean().reset_index()
        return agg_df

    # ---------------- 可视化 ----------------
    def plot_metric_by_time(self, time_unit='hour', metrics=None, kind='bar',
                        root_path=None, save=False):
        """
        可视化指定时间单位的互动指标

        参数:
        ----------
        time_unit : str
            'hour', 'weekday', 'date', 'month'
        metrics : list[str]
            互动指标列表，例如 ['liked_count', 'comment_count']
        kind : str
            'bar' 或 'line'
        root_path : str | None
            保存图片的根路径，当 save=True 时必须提供
        save : bool
            是否保存图像，为 True 时自动保存每个 metric 的图像
        """

        if metrics is None:
            metrics = ['total_engagement']
        if save and not root_path:
            raise ValueError("当 save=True 时，必须提供 root_path。")

        agg_df = self.aggregate_by_time(time_unit, metrics)

        for metric in metrics:
            plt.figure(figsize=(10, 5))

            if kind == 'bar':
                sns.barplot(data=agg_df, x=time_unit, y=metric, alpha=0.7)
            elif kind == 'line':
                sns.lineplot(data=agg_df, x=time_unit, y=metric, marker='o')
            else:
                raise ValueError("kind 必须是 'bar' 或 'line'")

            plt.title(f'{metric} vs {time_unit}')
            plt.xlabel(time_unit)
            plt.ylabel(metric)
            plt.tight_layout()

            # 如果需要保存
            if save:
                os.makedirs(root_path, exist_ok=True)
                file_path = os.path.join(root_path, f"{metric}.png")
                plt.savefig(file_path)
                print(f"✅ 已保存图像: {file_path}")

            # plt.show()
            plt.close()

    # ---------------- 相关性分析 ----------------
    def correlation_analysis(self, columns):
        """
        计算指定列之间的相关性
        :param columns: 列名列表
        :return: 相关性矩阵
        """
        corr = self.df[columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()
        return corr

if __name__ == "__main__":
    # 通用信息
    db_path = "/Users/charlottey./develop/MediaCrawler/database/sqlite_tables.db"
    target_table = "douyin_aweme"
    output_csv_path = "/Users/charlottey./develop/MediaCrawler/database"
    handler = SQLiteHandler(db_path)

    # 表名
    table_names = handler.list_tables()
    print("数据表基本信息：", len(table_names), table_names)

    # 抖音表信息
    dy_table_df = handler.get_df(target_table)
    print("抖音表例子：", dy_table_df.head(5))
    print("抖音表列名：", handler.get_all_column_names(target_table))

    # 目标like count区间
    like_df = handler.filter_by_range(dy_table_df, "liked_count", 100, 1000)
    like_df = handler.sort_df(like_df, "liked_count", ascending=False)
    print("目标like count区间：", like_df.head(5))
    handler.export_to_csv(like_df, os.path.join(output_csv_path, "dy_like.csv"))
    
    da = DataAnalyzer(dy_table_df)
    metrics_to_analyze = ["total_engagement", "liked_count", "comment_count", "share_count", "collected_count"]
    da.plot_metric_by_time(metrics=metrics_to_analyze, root_path=output_csv_path, save=True)
    # da.correlation_analysis(columns=["hour", "liked_count"])
    handler.close()