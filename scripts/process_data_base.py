import sqlite3
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import re
import numpy as np
from scipy.stats import pearsonr

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

    # ---------------- 文本分析 ----------------
    def add_text_stats(self, text_col='nickname'):
        """
        统计文本信息：总字符数、中文字符数、emoji数量、是否包含emoji
        并添加到df新列
        """
        def is_emoji(s):
            return any(char in emoji.EMOJI_DATA for char in s)

        def count_emoji(s):
            return sum(char in emoji.EMOJI_DATA for char in s)

        def count_chinese(s):
            return len(re.findall(r'[\u4e00-\u9fff]', s))

        self.df[f'{text_col}_total_chars'] = self.df[text_col].astype(str).apply(len)
        self.df[f'{text_col}_chinese_chars'] = self.df[text_col].astype(str).apply(count_chinese)
        self.df[f'{text_col}_emoji_count'] = self.df[text_col].astype(str).apply(count_emoji)
        self.df[f'{text_col}_has_emoji'] = self.df[f'{text_col}_emoji_count'] > 0

        return self.df

    # ---------------- 互动分析 ----------------
    def aggregate_by_time(self, time_unit='hour', metrics=['total_engagement'], agg_func='median'):
        """
        按时间单位统计互动指标均值
        :param time_unit: 'hour', 'weekday', 'date', 'month'
        :param metrics: 要分析的互动指标列表
        :return: 汇总 DataFrame
        """
        if time_unit not in ['hour', 'weekday', 'date', 'month']:
            raise ValueError("time_unit 必须是 'hour','weekday','date','month'")
        if agg_func == 'mean':
            agg_df = self.df.groupby(time_unit)[metrics].mean().reset_index()
        elif agg_func == 'sum':
            agg_df = self.df.groupby(time_unit)[metrics].sum().reset_index()
        elif agg_func == 'median':
            agg_df = self.df.groupby(time_unit)[metrics].median().reset_index()
        else:
            raise ValueError("agg_func 错误")
        return agg_df

    # ---------------- 可视化 ----------------
    def plot_metric_by_time(self, time_unit='hour', metrics=None, kind='bar',
                        root_path=None, save=False, agg_func='median'):
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

        agg_df = self.aggregate_by_time(time_unit, metrics, agg_func)

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
                file_path = os.path.join(root_path, f"{metric}_{agg_func}.png")
                plt.savefig(file_path)
                print(f"✅ 已保存图像: {file_path}")

            # plt.show()
            plt.close()

    def plot_bin_counts(self, col, start, end, step, save=False, save_path=None, prefix=None):
        """
        按区间统计指定列的个数，并绘图
        :param col: 列名
        :param start: 区间起始值
        :param end: 区间结束值
        :param step: 步长
        :param save: 是否保存图片
        :param save_path: 保存路径
        :param prefix: 文件名前缀
        """

        if save and not save_path:
            raise ValueError("当 save=True 时，必须提供 save_path。")

        # 生成区间
        bins = np.arange(start, end + step, step)
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

        # 统计频数
        df_cut = pd.cut(self.df[col], bins=bins, right=False, labels=labels)
        counts = df_cut.value_counts().sort_index()

        # 绘图
        plt.figure(figsize=(max(len(labels)*0.8, 8), 4))
        counts.plot(kind='bar', color='skyblue')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Count of {col} in bins')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 保存图片
        if save:
            os.makedirs(save_path, exist_ok=True)
            file_name = f"{prefix}_{col}_bin_counts.png" if prefix else f"{col}_bin_counts.png"
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, dpi=800)
            print(f"✅ 已保存图像: {file_path}")

        # plt.show()
        plt.close()
        return counts


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
        # plt.show()
        plt.close()
        return corr
    
    def correlation_analysis_matrix(self, targets, others, save=False, save_path=None, prefix=None):
        """
        多目标列与其他列的相关性分析
        :param targets: 目标列列表
        :param others: 其他列列表
        :return: DataFrame，每行是目标列与其他列的相关性
        """
        result = pd.DataFrame(index=targets, columns=others, dtype=float)
        if save and not save_path:
            raise ValueError("当 save=True 时，必须提供 root_path。")
        for target in targets:
            for col in others:
                result.loc[target, col] = self.df[[target, col]].corr().iloc[0, 1]
        
        # 可视化
        plt.figure(figsize=(len(others)*4, len(targets)*2))
        sns.heatmap(result.astype(float), annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
        plt.title(f"Correlation: targets vs others")
        x_labels = [label if len(label) <= 10 else label[:10] + '\n' + label[10:] for label in others]
        plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=0, ha='right')
        plt.yticks(rotation=0) 
        # 如果需要保存
        if save:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"{prefix}_correlation.png")
            plt.savefig(file_path, dpi=800)
            print(f"✅ 已保存图像: {file_path}")

        # plt.show()
        plt.close()
        return result

    def scatter_with_regression_save(self, x_col, y_col, save=False, save_path=None, prefix=None):
        """
        绘制散点图 + 回归线，显示相关性系数，并支持保存
        :param x_col: X轴列名
        :param y_col: Y轴列名
        :param save: 是否保存图像
        :param save_path: 保存路径
        :param prefix: 文件名前缀
        :return: 相关性系数
        """

        if save and not save_path:
            raise ValueError("当 save=True 时，必须提供 save_path。")

        x = self.df[x_col]
        y = self.df[y_col]

        # 计算 Pearson 相关系数
        corr_coef, _ = pearsonr(x, y)

        # 绘图
        plt.figure(figsize=(6, 4))
        sns.regplot(x=x, y=y, scatter_kws={'s':50}, line_kws={'color':'red'})
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter plot of {x_col} vs {y_col}\nPearson r = {corr_coef:.3f}')

        # 将相关性系数显示在图右上角
        plt.text(0.95, 0.05, f'r = {corr_coef:.3f}', transform=plt.gca().transAxes,
                ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # 保存图像
        if save:
            os.makedirs(save_path, exist_ok=True)
            file_name = f"{prefix}_{x_col}_vs_{y_col}.png" if prefix else f"{x_col}_vs_{y_col}.png"
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, dpi=800, bbox_inches='tight')
            print(f"✅ 已保存图像: {file_path}")

        # plt.show()
        plt.close()

        return corr_coef



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
    # 互动率中位数与发布时间关系
    da.plot_metric_by_time(metrics=metrics_to_analyze, root_path=output_csv_path, save=True, agg_func="median")
    da.correlation_analysis(columns=["hour", "liked_count"])
    
    # 提取nickname特征
    text_col = "nickname"
    da.add_text_stats(text_col)
    da.correlation_analysis_matrix(targets=["liked_count", "comment_count", "share_count", "collected_count"], 
                                   others=[f'{text_col}_total_chars', f'{text_col}_chinese_chars', f'{text_col}_emoji_count', f'{text_col}_has_emoji'],
                                   save=True, save_path=output_csv_path, prefix=text_col)

    # 提取desc特征
    text_col = "desc"
    da.add_text_stats(text_col)
    da.correlation_analysis_matrix(targets=["liked_count", "comment_count", "share_count", "collected_count"], 
                                   others=[f'{text_col}_total_chars', f'{text_col}_chinese_chars', f'{text_col}_emoji_count', f'{text_col}_has_emoji'],
                                   save=True, save_path=output_csv_path, prefix=text_col)
    
    # 自由对比
    da.scatter_with_regression_save(x_col="hour", y_col="liked_count", save=True, save_path=output_csv_path)
    
    # 单列统计
    da.plot_bin_counts(col='hour', start=0, end=24, step=1, save=True, save_path=output_csv_path)
    handler.close()