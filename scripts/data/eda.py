import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Dataset paths
USERS_PATH = '/Users/sachidhoka/Desktop/migraine_users.csv'
DAILY_LOGS_PATH = '/Users/sachidhoka/Desktop/migraine_daily_records.csv'

# Output directories
OUTPUT_DIR = 'data/processed/'
PLOTS_DIR = 'data/processed/milestone1_plots/'

class eda:
    """Main class for Milestone 1 data validation and quality assessment"""
    
    def __init__(self, users_path, daily_logs_path):
        self.users_path = users_path
        self.daily_logs_path = daily_logs_path
        self.users_df = None
        self.daily_logs_df = None
        self.quality_report = {
            'summary': {},
            'issues': [],
            'validations': {},
            'statistics': {}
        }
        
    def load_data(self):
        """Load Users and Daily Logs tables"""
        print("=" * 80)
        print("STEP 1: Loading Data")
        print("=" * 80)
        
        try:
            self.users_df = pd.read_csv(self.users_path)
            self.daily_logs_df = pd.read_csv(self.daily_logs_path)
            
            print(f"✓ Users table loaded: {len(self.users_df)} rows")
            print(f"✓ Daily logs table loaded: {len(self.daily_logs_df)} rows")
            
            # Convert date column to datetime (column is named 'date' in your dataset)
            self.daily_logs_df['log_date'] = pd.to_datetime(self.daily_logs_df['date'])
            
            self.quality_report['summary']['users_count'] = len(self.users_df)
            self.quality_report['summary']['logs_count'] = len(self.daily_logs_df)
            
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.quality_report['issues'].append(f"Data loading failed: {str(e)}")
            return False
    
    def generate_summary_statistics(self):
        """Generate summary statistics for both tables"""
        print("\n" + "=" * 80)
        print("STEP 2: Generating Summary Statistics")
        print("=" * 80)
        
        # Users table statistics
        print("\n--- USERS TABLE ---")
        print(f"Total users: {len(self.users_df)}")
        print(f"Unique user IDs: {self.users_df['user_id'].nunique()}")
        print(f"\nColumns: {list(self.users_df.columns)}")
        print(f"\nData types:\n{self.users_df.dtypes}")
        print(f"\nMissing values:\n{self.users_df.isnull().sum()}")
        
        # Daily logs statistics
        print("\n--- DAILY LOGS TABLE ---")
        print(f"Total log entries: {len(self.daily_logs_df)}")
        print(f"Unique users in logs: {self.daily_logs_df['user_id'].nunique()}")
        print(f"Date range: {self.daily_logs_df['log_date'].min()} to {self.daily_logs_df['log_date'].max()}")
        print(f"\nColumns: {list(self.daily_logs_df.columns)}")
        
        # Missing value percentages
        missing_pct = (self.daily_logs_df.isnull().sum() / len(self.daily_logs_df) * 100).round(2)
        print(f"\nMissing value percentages:\n{missing_pct[missing_pct > 0]}")
        
        # Per-user statistics
        user_log_counts = self.daily_logs_df.groupby('user_id').size()
        print(f"\n--- PER-USER LOG STATISTICS ---")
        print(f"Mean logs per user: {user_log_counts.mean():.1f}")
        print(f"Median logs per user: {user_log_counts.median():.1f}")
        print(f"Min logs per user: {user_log_counts.min()}")
        print(f"Max logs per user: {user_log_counts.max()}")
        
        # Store in report
        self.quality_report['statistics']['user_log_counts'] = user_log_counts.describe().to_dict()
        self.quality_report['statistics']['missing_percentages'] = missing_pct.to_dict()
        
    def check_data_quality(self):
        """Validate data quality and check for impossible values"""
        print("\n" + "=" * 80)
        print("STEP 3: Data Quality Validation")
        print("=" * 80)
        
        issues = []
        
        # Check for duplicate user_ids in users table
        duplicate_users = self.users_df['user_id'].duplicated().sum()
        if duplicate_users > 0:
            issues.append(f"Found {duplicate_users} duplicate user IDs in Users table")
        else:
            print("✓ No duplicate user IDs in Users table")
        
        # Check referential integrity
        logs_users = set(self.daily_logs_df['user_id'].unique())
        users_users = set(self.users_df['user_id'].unique())
        orphaned_logs = logs_users - users_users
        
        if orphaned_logs:
            issues.append(f"Found {len(orphaned_logs)} user IDs in logs without corresponding Users entry")
            print(f"✗ {len(orphaned_logs)} orphaned user IDs in logs")
        else:
            print("✓ All log entries have corresponding user records")
        
        self.quality_report['validations']['referential_integrity'] = len(orphaned_logs) == 0
        
        # Check for impossible values
        print("\n--- CHECKING FOR IMPOSSIBLE VALUES ---")
        
        # Sleep hours (should be 0-24)
        invalid_sleep = self.daily_logs_df[
            (self.daily_logs_df['sleep_hours'] < 0) | 
            (self.daily_logs_df['sleep_hours'] > 24)
        ]
        if len(invalid_sleep) > 0:
            issues.append(f"Found {len(invalid_sleep)} records with invalid sleep_hours")
            print(f"✗ {len(invalid_sleep)} records with sleep_hours outside 0-24 range")
        else:
            print("✓ All sleep_hours values are valid (0-24)")
        
        # Pain intensity (should be 0-10)
        invalid_pain = self.daily_logs_df[
            (self.daily_logs_df['pain_intensity'] < 0) | 
            (self.daily_logs_df['pain_intensity'] > 10)
        ]
        if len(invalid_pain) > 0:
            issues.append(f"Found {len(invalid_pain)} records with invalid pain_intensity")
            print(f"✗ {len(invalid_pain)} records with pain_intensity outside 0-10 range")
        else:
            print("✓ All pain_intensity values are valid (0-10)")
        
        # Stress level (should be 0-10)
        invalid_stress = self.daily_logs_df[
            (self.daily_logs_df['stress_level'] < 0) | 
            (self.daily_logs_df['stress_level'] > 10)
        ]
        if len(invalid_stress) > 0:
            issues.append(f"Found {len(invalid_stress)} records with invalid stress_level")
            print(f"✗ {len(invalid_stress)} records with stress_level outside 0-10 range")
        else:
            print("✓ All stress_level values are valid (0-10)")
        
        # Water glasses (should be >= 0, typically < 20)
        invalid_water = self.daily_logs_df[self.daily_logs_df['water_glasses'] < 0]
        if len(invalid_water) > 0:
            issues.append(f"Found {len(invalid_water)} records with negative water_glasses")
            print(f"✗ {len(invalid_water)} records with negative water_glasses")
        else:
            print("✓ All water_glasses values are non-negative")
        
        # Caffeine cups (should be >= 0, typically < 20)
        invalid_caffeine = self.daily_logs_df[self.daily_logs_df['caffeine_cups'] < 0]
        if len(invalid_caffeine) > 0:
            issues.append(f"Found {len(invalid_caffeine)} records with negative caffeine_cups")
            print(f"✗ {len(invalid_caffeine)} records with negative caffeine_cups")
        else:
            print("✓ All caffeine_cups values are non-negative")
        
        self.quality_report['issues'].extend(issues)
        
    def validate_temporal_coverage(self):
        """Verify each user has logs spanning at least 30 consecutive days"""
        print("\n" + "=" * 80)
        print("STEP 4: Temporal Coverage Validation")
        print("=" * 80)
        
        user_date_ranges = []
        users_with_gaps = []
        
        for user_id in self.daily_logs_df['user_id'].unique():
            user_logs = self.daily_logs_df[self.daily_logs_df['user_id'] == user_id].sort_values('log_date')
            
            if len(user_logs) == 0:
                continue
            
            min_date = user_logs['log_date'].min()
            max_date = user_logs['log_date'].max()
            date_span = (max_date - min_date).days + 1
            actual_logs = len(user_logs)
            
            # Check for gaps
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            missing_dates = len(expected_dates) - actual_logs
            
            user_date_ranges.append({
                'user_id': user_id,
                'start_date': min_date,
                'end_date': max_date,
                'date_span_days': date_span,
                'actual_logs': actual_logs,
                'missing_dates': missing_dates,
                'has_30_days': date_span >= 30
            })
            
            if missing_dates > 0:
                users_with_gaps.append(user_id)
        
        coverage_df = pd.DataFrame(user_date_ranges)
        
        # Summary statistics
        users_with_30_days = coverage_df['has_30_days'].sum()
        total_users = len(coverage_df)
        coverage_percentage = (users_with_30_days / total_users * 100) if total_users > 0 else 0
        
        print(f"\nTemporal Coverage Summary:")
        print(f"Total users analyzed: {total_users}")
        print(f"Users with 30+ day span: {users_with_30_days} ({coverage_percentage:.1f}%)")
        print(f"Users with date gaps: {len(users_with_gaps)}")
        print(f"\nDate span statistics:")
        print(coverage_df['date_span_days'].describe())
        
        self.quality_report['validations']['temporal_coverage'] = {
            'users_with_30_days': users_with_30_days,
            'total_users': total_users,
            'coverage_percentage': coverage_percentage,
            'users_with_gaps': len(users_with_gaps)
        }
        
        # Save detailed coverage data
        coverage_df.to_csv(f'{OUTPUT_DIR}user_temporal_coverage.csv', index=False)
        print(f"\n✓ Detailed coverage data saved to {OUTPUT_DIR}user_temporal_coverage.csv")
        
        return coverage_df
    
    def analyze_migraine_rates(self):
        """Validate migraine occurrence rates"""
        print("\n" + "=" * 80)
        print("STEP 5: Migraine Occurrence Analysis")
        print("=" * 80)
        
        # Overall migraine rate
        total_logs = len(self.daily_logs_df)
        migraine_logs = self.daily_logs_df['migraine_occurred'].sum()
        overall_rate = (migraine_logs / total_logs * 100) if total_logs > 0 else 0
        
        print(f"\nOverall Migraine Statistics:")
        print(f"Total log entries: {total_logs}")
        print(f"Migraine occurrences: {migraine_logs}")
        print(f"Overall migraine rate: {overall_rate:.2f}%")
        
        # Per-user migraine rates
        user_migraine_stats = self.daily_logs_df.groupby('user_id').agg({
            'migraine_occurred': ['sum', 'count', 'mean']
        }).reset_index()
        user_migraine_stats.columns = ['user_id', 'migraine_count', 'total_logs', 'migraine_rate']
        user_migraine_stats['migraine_rate'] = user_migraine_stats['migraine_rate'] * 100
        
        print(f"\nPer-User Migraine Rate Statistics:")
        print(user_migraine_stats['migraine_rate'].describe())
        
        # Check if rates are realistic (5-20%)
        if 5 <= overall_rate <= 20:
            print(f"\n✓ Overall migraine rate ({overall_rate:.2f}%) is within realistic range (5-20%)")
            self.quality_report['validations']['migraine_rate_realistic'] = True
        else:
            print(f"\n✗ Overall migraine rate ({overall_rate:.2f}%) is outside expected range (5-20%)")
            self.quality_report['validations']['migraine_rate_realistic'] = False
            self.quality_report['issues'].append(f"Migraine rate {overall_rate:.2f}% outside realistic range")
        
        self.quality_report['statistics']['migraine_rates'] = {
            'overall_rate': overall_rate,
            'total_migraines': int(migraine_logs),
            'user_rate_mean': float(user_migraine_stats['migraine_rate'].mean()),
            'user_rate_std': float(user_migraine_stats['migraine_rate'].std())
        }
        
        # Save per-user statistics
        user_migraine_stats.to_csv(f'{OUTPUT_DIR}user_migraine_statistics.csv', index=False)
        print(f"\n✓ Per-user statistics saved to {OUTPUT_DIR}user_migraine_statistics.csv")
        
        return user_migraine_stats
    
    def analyze_correlations(self):
        """Inspect correlations between weather variables and migraine occurrence"""
        print("\n" + "=" * 80)
        print("STEP 6: Correlation Analysis")
        print("=" * 80)
        
        # Select numeric columns for correlation
        numeric_cols = self.daily_logs_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove user_id from correlation analysis
        if 'user_id' in numeric_cols:
            numeric_cols.remove('user_id')
        
        corr_matrix = self.daily_logs_df[numeric_cols].corr()
        
        # Focus on migraine correlations
        if 'migraine_occurred' in corr_matrix.columns:
            migraine_corr = corr_matrix['migraine_occurred'].sort_values(ascending=False)
            print("\nCorrelations with Migraine Occurrence:")
            print(migraine_corr)
            
            # Identify top positive and negative correlators
            top_positive = migraine_corr[migraine_corr > 0].iloc[1:6]  # Skip self-correlation
            top_negative = migraine_corr[migraine_corr < 0].iloc[-5:]
            
            print(f"\nTop 5 Positive Correlators:")
            print(top_positive)
            print(f"\nTop 5 Negative Correlators:")
            print(top_negative)
            
            self.quality_report['statistics']['migraine_correlations'] = {
                'top_positive': top_positive.to_dict(),
                'top_negative': top_negative.to_dict()
            }
        
        return corr_matrix
    
    def create_visualizations(self, coverage_df, user_stats, corr_matrix):
        """Generate all required visualizations"""
        print("\n" + "=" * 80)
        print("STEP 7: Creating Visualizations")
        print("=" * 80)
        
        import os
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # 1. Distribution plots for numeric columns
        print("\nGenerating distribution plots...")
        numeric_cols = ['sleep_hours', 'sleep_quality', 'stress_level', 'pain_intensity', 
                       'water_glasses', 'caffeine_cups', 'temperature_c', 
                       'humidity_percent', 'barometric_pressure', 'air_quality_index',
                       'duration_hours', 'exercise_duration', 'screen_time']
        
        # Filter to only existing columns
        available_numeric = [col for col in numeric_cols if col in self.daily_logs_df.columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(available_numeric[:9]):
            if col in self.daily_logs_df.columns:
                self.daily_logs_df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOTS_DIR}numeric_distributions.png")
        
        # 2. Correlation heatmap
        print("Generating correlation heatmap...")
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOTS_DIR}correlation_heatmap.png")
        
        # 3. Migraine occurrence rate by user
        print("Generating migraine rate visualization...")
        plt.figure(figsize=(12, 6))
        user_stats_sorted = user_stats.sort_values('migraine_rate', ascending=False).head(20)
        plt.bar(range(len(user_stats_sorted)), user_stats_sorted['migraine_rate'])
        plt.axhline(y=5, color='g', linestyle='--', label='Min Expected (5%)')
        plt.axhline(y=20, color='r', linestyle='--', label='Max Expected (20%)')
        plt.xlabel('User (Top 20)')
        plt.ylabel('Migraine Rate (%)')
        plt.title('Per-User Migraine Occurrence Rates (Top 20 Users)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}user_migraine_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOTS_DIR}user_migraine_rates.png")
        
        # 4. Temporal coverage visualization
        print("Generating temporal coverage visualization...")
        plt.figure(figsize=(12, 6))
        plt.hist(coverage_df['date_span_days'], bins=30, edgecolor='black')
        plt.axvline(x=30, color='r', linestyle='--', linewidth=2, label='30-day threshold')
        plt.xlabel('Date Span (days)')
        plt.ylabel('Number of Users')
        plt.title('Distribution of User Date Span Coverage')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}temporal_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOTS_DIR}temporal_coverage.png")
        
        # 5. Sample user timelines
        print("Generating sample user timelines...")
        sample_users = self.daily_logs_df['user_id'].unique()[:5]
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        for idx, user_id in enumerate(sample_users):
            user_data = self.daily_logs_df[self.daily_logs_df['user_id'] == user_id].sort_values('log_date')
            
            axes[idx].plot(user_data['log_date'], user_data['stress_level'], 
                          label='Stress Level', alpha=0.7)
            
            # Mark migraine days
            migraine_days = user_data[user_data['migraine_occurred'] == 1]
            axes[idx].scatter(migraine_days['log_date'], 
                            migraine_days['stress_level'], 
                            color='red', s=100, marker='x', 
                            label='Migraine Day', zorder=5)
            
            axes[idx].set_ylabel('Stress Level')
            axes[idx].set_title(f'User {user_id} Timeline')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}sample_user_timelines.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOTS_DIR}sample_user_timelines.png")
        
        print(f"\n✓ All visualizations saved to {PLOTS_DIR}")
    
    def generate_quality_report(self):
        """Create comprehensive data quality report"""
        print("\n" + "=" * 80)
        print("STEP 8: Generating Data Quality Report")
        print("=" * 80)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MIGRAINEMAMBA - MILESTONE 1 DATA QUALITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary Section
        report_lines.append("\n" + "=" * 80)
        report_lines.append("1. DATA SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Users table rows: {self.quality_report['summary']['users_count']}")
        report_lines.append(f"Daily logs table rows: {self.quality_report['summary']['logs_count']}")
        
        # Validation Results
        report_lines.append("\n" + "=" * 80)
        report_lines.append("2. VALIDATION RESULTS")
        report_lines.append("=" * 80)
        
        for key, value in self.quality_report['validations'].items():
            if isinstance(value, bool):
                status = "✓ PASS" if value else "✗ FAIL"
                report_lines.append(f"{key}: {status}")
            elif isinstance(value, dict):
                report_lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"  {sub_key}: {sub_value}")
        
        # Issues Found
        report_lines.append("\n" + "=" * 80)
        report_lines.append("3. ISSUES IDENTIFIED")
        report_lines.append("=" * 80)
        
        if self.quality_report['issues']:
            for issue in self.quality_report['issues']:
                report_lines.append(f"• {issue}")
        else:
            report_lines.append("✓ No critical issues identified")
        
        # Statistics
        report_lines.append("\n" + "=" * 80)
        report_lines.append("4. KEY STATISTICS")
        report_lines.append("=" * 80)
        
        if 'migraine_rates' in self.quality_report['statistics']:
            mr = self.quality_report['statistics']['migraine_rates']
            report_lines.append(f"\nMigraine Statistics:")
            report_lines.append(f"  Overall rate: {mr['overall_rate']:.2f}%")
            report_lines.append(f"  Total migraine events: {mr['total_migraines']}")
            report_lines.append(f"  Mean user rate: {mr['user_rate_mean']:.2f}%")
        
        # Success Criteria Check
        report_lines.append("\n" + "=" * 80)
        report_lines.append("5. SUCCESS CRITERIA EVALUATION")
        report_lines.append("=" * 80)
        
        criteria_met = []
        criteria_failed = []
        
        # Check each criterion
        if self.quality_report['validations'].get('referential_integrity', False):
            criteria_met.append("Zero broken user_id references between tables")
        else:
            criteria_failed.append("Zero broken user_id references between tables")
        
        if self.quality_report['validations'].get('migraine_rate_realistic', False):
            criteria_met.append("Migraine occurrence rate between 5-20%")
        else:
            criteria_failed.append("Migraine occurrence rate between 5-20%")
        
        temporal = self.quality_report['validations'].get('temporal_coverage', {})
        if temporal.get('coverage_percentage', 0) >= 80:
            criteria_met.append("At least 80% of users have 30+ days of continuous logs")
        else:
            criteria_failed.append("At least 80% of users have 30+ days of continuous logs")
        
        if not any('impossible values' in issue.lower() for issue in self.quality_report['issues']):
            criteria_met.append("No impossible values found")
        else:
            criteria_failed.append("No impossible values found")
        
        report_lines.append("\n✓ CRITERIA MET:")
        for criterion in criteria_met:
            report_lines.append(f"  • {criterion}")
        
        if criteria_failed:
            report_lines.append("\n✗ CRITERIA NOT MET:")
            for criterion in criteria_failed:
                report_lines.append(f"  • {criterion}")
        
        # Recommendations
        report_lines.append("\n" + "=" * 80)
        report_lines.append("6. RECOMMENDATIONS")
        report_lines.append("=" * 80)
        
        if len(criteria_failed) == 0:
            report_lines.append("✓ All success criteria met. Data is ready for Milestone 2.")
        else:
            report_lines.append("⚠ Some criteria not met. Review issues before proceeding.")
            report_lines.append("\nRecommended actions:")
            for issue in self.quality_report['issues'][:5]:
                report_lines.append(f"  • Address: {issue}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Write report to file
        report_text = "\n".join(report_lines)
        with open(f'{OUTPUT_DIR}milestone1_data_quality_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Full report saved to {OUTPUT_DIR}milestone1_data_quality_report.txt")
    
    def run_complete_validation(self):
        """Execute all validation steps"""
        print("\n" + "=" * 80)
        print("MIGRAINEMAMBA - MILESTONE 1: DATA DISCOVERY & VALIDATION")
        print("Starting Complete Validation Pipeline")
        print("=" * 80 + "\n")
        
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # Execute all steps
        if not self.load_data():
            print("\n✗ Validation failed at data loading step")
            return
        
        self.generate_summary_statistics()
        self.check_data_quality()
        coverage_df = self.validate_temporal_coverage()
        user_stats = self.analyze_migraine_rates()
        corr_matrix = self.analyze_correlations()
        self.create_visualizations(coverage_df, user_stats, corr_matrix)
        self.generate_quality_report()
        
        print("\n" + "=" * 80)
        print("MILESTONE 1 VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nDeliverables generated:")
        print(f"  ✓ Data quality report: {OUTPUT_DIR}milestone1_data_quality_report.txt")
        print(f"  ✓ User statistics: {OUTPUT_DIR}user_migraine_statistics.csv")
        print(f"  ✓ Temporal coverage: {OUTPUT_DIR}user_temporal_coverage.csv")
        print(f"  ✓ Visualizations: {PLOTS_DIR}")
        print("\n" + "=" * 80 + "\n")


def main():
    """Main execution function"""
    validator = eda(USERS_PATH, DAILY_LOGS_PATH)
    validator.run_complete_validation()


if __name__ == "__main__":
    main()