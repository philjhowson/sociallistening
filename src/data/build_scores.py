import pickle
import pandas as pd
import shared_functions

def score():

    summary_scores = {
        'General Experience with Extended Warranty Services': {
            'Pain Points': {
                'Lack of Transparency in Coverage Terms': 13.99,
                'Customer Service Quality and Resolution Delays': 13.42,
                'Unmet Expectations Around Repair Speed': 13.33,
                'Difficulties Transferring or Cancelling Warranties': 12.84,
                'Upfront Cost Discourages Many, Especially for Low-Income Buyers': 12.66
            },
            'Drivers': {
                'High Value from One-Time Coverage Success': 13.84,
                'Peace of Mind and Risk Reduction': 13.66,
                'Bundled Support and Priority Service Access': 13.46,
                'Social Proof from Friends and Online Communities': 13.12,
                'Long-Term Use Plans Justify Cost': 12.95
            }
        },
        'Role of Experience in Customer Satisfaction and Brand Impressions': {
            'Pain Points': {
                'Poor After-Sales Experience Damages Long-Term Trust': 14.16,
                'Inconsistent Service Across Locations': 13.84,
                'Negative Employee Attitudes Undermine Support Quality': 13.59,
                'Difficulty Escalating Unresolved Cases': 13.44,
                'Lack of Proactive Communication': 13.11
            },
            'Drivers': {
                'Positive Service Turns Customers into Promoters': 13.94,
                'Seamless Omnichannel Support Enhances Perceived Professionalism': 13.76,
                'Friendly and Empowered Staff Leave a Strong Brand Impression': 13.59,
                'Perceived Fairness and Transparency in Decisions': 13.37,
                ' Warranty Experience Reinforces Brand Reliability': 13.12
            }
        },
        'Willingness to Pay (WTP)': {
            'Pain Points': {
                'High Prices Discourage Rational Buyers': 14.34,
                'Bundled Warranties Without Consent Create Resentment': 14.11,
                'Low Trust in Claim Success Reduces WTP': 13.98,
                'Lack of Flexible Plans for Low-Risk Users': 13.68,
                'Perception That Warranties Are Profit Tools, Not Support': 13.51
            },
            'Drivers': {
                'Clear Past Value Drives Future Willingness': 14.29,
                'Willingness to Pay Increases with Product Value and Risk': 14.11,
                'Monthly Subscription Options Make Warranties More Appealing': 13.91,
                'Strong Brand Trust Increases WTP': 13.57,
                'Peer Recommendations and Social Norms Boost Confidence': 13.46
            }
        },
        'General Sentiment on Warranties and Extended Warranties': {
            'Pain Points': {
                'Language Around "Lifetime Warranty" Feels Deceptive': 13.86,
                'Manufacturer Warranty Gaps Drive Warranty Fatigue': 13.69,
                'Inflexibility of Warranty Terms for Edge Cases': 13.48,
                'Over-Reliance on Third-Party Warranty Providers': 13.31,
                'Cultural Differences in Warranty Norms': 13.14
            },
            'Drivers': {
                'Extended Warranties Offer Psychological Comfort': 13.71,
                'Warranties Help Buyers Take Risks on New Technology': 13.54,
                'Lifetime or Long-Term Coverage Enhances Brand Trust': 13.42,
                'Straightforward Claims Build Loyalty': 13.26,
                'Transparent Terms Reduce Fear and Increase Conversion': 13.09
            }
        }
    }

    shared_functions.safe_saver(summary_scores, 'data/processed/pain_points_drivers_scores.pkl')

    print(summary_scores['General Experience with Extended Warranty Services'])
    print(summary_scores['Role of Experience in Customer Satisfaction and Brand Impressions'])
    print(summary_scores['Willingness to Pay (WTP)'])
    print(summary_scores['General Sentiment on Warranties and Extended Warranties'])

    print(summary_scores['General Sentiment on Warranties and Extended Warranties']['Pain Points'])

if __name__ == '__main__':
    score()