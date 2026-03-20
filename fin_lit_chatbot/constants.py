from __future__ import annotations

RISK_QUIZ = {
	1: {
		"dimension": "willingness",
		"question": "Suppose you want an investment account to stay at or above a certain value. After a market drop, it falls below that level. What would you be more likely to do?",
		"options": {
			"A": "Stay calm and keep holding the investment.",
			"B": "Sell quickly to avoid the chance of further losses.",
		},
		"score": {"A": 1, "B": 0},
		"option_explanations": {
			"A": "This suggests greater comfort with short-term market swings and a willingness to wait for possible recovery.",
			"B": "This suggests lower emotional tolerance for volatility and a stronger desire to avoid losses immediately.",
		},
		"interpretation_tags": {
			"A": "higher_willingness_for_risk",
			"B": "lower_willingness_for_risk",
		},
		"followup_tip": "This question is mainly about emotional reaction to losses, not whether one answer is universally correct.",
	},
	2: {
		"dimension": "willingness",
		"question": "Which would you prefer?",
		"options": {
			"A": "A risky opportunity that could lead to a much larger gain, but could also result in no gain.",
			"B": "A guaranteed smaller gain.",
		},
		"score": {"A": 1, "B": 0},
		"option_explanations": {
			"A": "This suggests more comfort with uncertainty and a greater willingness to accept risk in exchange for higher upside.",
			"B": "This suggests a preference for certainty and stability, even if the potential reward is lower.",
		},
		"interpretation_tags": {
			"A": "higher_willingness_for_risk",
			"B": "lower_willingness_for_risk",
		},
		"followup_tip": "This question is about risk preference tradeoffs between certainty and possible higher returns.",
	},
	3: {
		"dimension": "capacity",
		"question": "Think about the money you are investing for a goal. Which description fits that goal better?",
		"options": {
			"A": "It is important, but missing it would not seriously damage my financial situation.",
			"B": "It is essential, and falling short would seriously affect me.",
		},
		"score": {"A": 1, "B": 0},
		"option_explanations": {
			"A": "This suggests you may have more capacity to tolerate investment risk because the goal is more flexible.",
			"B": "This suggests lower capacity for risk because the consequences of loss or underperformance are more serious.",
		},
		"interpretation_tags": {
			"A": "higher_capacity_for_risk",
			"B": "lower_capacity_for_risk",
		},
		"followup_tip": "This question is about how much financial damage you could absorb if things do not go well.",
	},
	4: {
		"dimension": "capacity",
		"question": "For a long-term investing goal such as retirement, which time horizon sounds closer to your situation?",
		"options": {
			"A": "The goal is still many years away.",
			"B": "The goal is relatively soon.",
		},
		"score": {"A": 1, "B": 0},
		"option_explanations": {
			"A": "A longer time horizon usually means more ability to ride out short-term volatility and recover from market declines.",
			"B": "A shorter time horizon usually means less room to recover from losses, so risk capacity is lower.",
		},
		"interpretation_tags": {
			"A": "higher_capacity_for_risk",
			"B": "lower_capacity_for_risk",
		},
		"followup_tip": "This question focuses on practical risk-bearing ability, not just personal comfort.",
	},
}

RISK_PROFILE_BANDS = [
	{
		"min_total": 0,
		"max_total": 1,
		"profile": "conservative",
		"summary": "Lower willingness and/or lower capacity for risk.",
	},
	{
		"min_total": 2,
		"max_total": 2,
		"profile": "moderately_conservative",
		"summary": "Some risk tolerance, but with meaningful caution.",
	},
	{
		"min_total": 3,
		"max_total": 3,
		"profile": "moderate",
		"summary": "Balanced willingness and capacity for some investment risk.",
	},
	{
		"min_total": 4,
		"max_total": 4,
		"profile": "moderately_aggressive",
		"summary": "Higher willingness and capacity for taking investment risk.",
	},
]

RISK_MISMATCH_INTERPRETATIONS = {
	"willingness_gt_capacity": "You may feel comfortable taking risk, but your financial situation or goals may not support as much risk.",
	"capacity_gt_willingness": "You may be financially able to take more risk than you feel emotionally comfortable taking.",
	"aligned": "Your emotional comfort and practical ability to take risk are relatively aligned.",
}

RISK_RESULT_TEMPLATES = {
	"conservative": {
		"headline": "You currently lean conservative.",
		"explanation": "You may prefer lower-risk approaches, or your current situation may not support taking much investment risk.",
		"next_steps": [
			"Learn about emergency funds",
			"Compare lower-volatility products",
			"Understand basic diversification",
		],
	},
	"moderately_conservative": {
		"headline": "You currently lean moderately conservative.",
		"explanation": "You may be open to some risk, but stability and downside protection are still important to you.",
		"next_steps": [
			"Compare bonds, diversified funds, and equities at a high level",
			"Learn risk vs return",
			"Understand time horizon and goal matching",
		],
	},
	"moderate": {
		"headline": "You currently look moderate.",
		"explanation": "You appear to have a reasonable mix of risk tolerance and risk capacity, though product suitability still depends on the exact goal.",
		"next_steps": [
			"Compare balanced investment approaches",
			"Learn diversification in more detail",
			"Explore how time horizon affects portfolio choices",
		],
	},
	"moderately_aggressive": {
		"headline": "You currently lean moderately aggressive.",
		"explanation": "You appear more comfortable with risk and may also have the capacity to bear more volatility, though risk control still matters.",
		"next_steps": [
			"Learn growth-oriented product categories",
			"Study diversification and concentration risk",
			"Understand drawdowns and long-term investing discipline",
		],
	},
}
