"""Generate a realistic synthetic employee review dataset (~66K reviews)."""
import random
import csv
from pathlib import Path

random.seed(42)

# Review templates by sentiment
NEGATIVE_TEMPLATES = [
    "The management here is terrible. They don't care about employees at all.",
    "Toxic work environment. Nobody respects your time or boundaries.",
    "Worst job I've ever had. The pay is awful and benefits are nonexistent.",
    "Constant layoffs and zero job security. Management lies about everything.",
    "The CEO is completely out of touch with what employees actually need.",
    "No work-life balance whatsoever. Expected to work weekends without pay.",
    "The culture is incredibly toxic. Backstabbing and politics everywhere.",
    "Training is nonexistent. You're thrown in without any support or guidance.",
    "Salary is way below market rate. They exploit junior employees.",
    "The building is falling apart. No investment in infrastructure or tools.",
    "HR is useless. Reported harassment and nothing happened.",
    "Micromanagement at its finest. Can't breathe without permission.",
    "The benefits package is a joke. Health insurance barely covers anything.",
    "Promotions are impossible unless you're best friends with management.",
    "The workload is unsustainable. People are burning out left and right.",
    "No diversity or inclusion efforts whatsoever. Very homogeneous leadership.",
    "The company promises growth but delivers stagnation. Dead-end role.",
    "Communication from leadership is nonexistent. No one knows what's going on.",
    "The turnover rate is insane. Good people leave every month.",
    "Disorganized chaos. No processes, no structure, no leadership.",
    "The company culture is all talk. Values mean nothing in practice.",
    "I was overworked and underpaid. My manager took credit for all my work.",
    "The office politics are unbearable. Can't focus on actual work.",
    "Technology stack is ancient. No willingness to modernize or innovate.",
    "The onboarding process is broken. Spent weeks confused and unsupported.",
    "Management plays favorites. Performance reviews are subjective and unfair.",
    "The company is sinking. Revenue is down and morale is at rock bottom.",
    "No transparency from leadership. Decisions are made behind closed doors.",
    "The work is repetitive and boring. No challenge or opportunity to grow.",
    "Benefits keep getting worse every year while workload keeps increasing.",
    "The company doesn't value employee feedback. Surveys lead to nothing.",
    "I dread going to work every day. The environment is that bad.",
    "The hiring process is a mess. Took 6 rounds of interviews for nothing.",
    "My manager had no idea what they were doing. Incompetent leadership.",
    "The company prioritizes profits over people in every decision.",
    "Long hours, low pay, zero appreciation. A recipe for burnout.",
    "The team is dysfunctional. No collaboration or teamwork at all.",
    "I was promised a raise that never came. Management broke every promise.",
    "The company has no vision. Just reacting to problems instead of planning.",
    "Work-life balance is a myth here. You're always on call.",
]

NEUTRAL_TEMPLATES = [
    "It's an okay place to work. Nothing special but gets the job done.",
    "The job is fine. Not great, not terrible. Just average.",
    "Decent place for a first job. Good enough to gain some experience.",
    "The work is straightforward. Nothing exciting but not boring either.",
    "Average salary and average benefits. Nothing to complain about really.",
    "The team is okay. Some people are nice, others not so much.",
    "It's a stable job. Not much growth opportunity but the pay is consistent.",
    "The office environment is decent. Could be better but could be worse.",
    "Management is average. They do their job but nothing exceptional.",
    "The workload is manageable most of the time. Busy seasons are tough.",
    "Benefits are standard for the industry. Nothing stands out.",
    "The company is fine for work-life balance. Not perfect but acceptable.",
    "Decent training programs. Could use more but what's there is useful.",
    "The technology we use is okay. Not cutting edge but functional.",
    "Communication could be better but it's not the worst I've experienced.",
    "The company culture is neutral. People are professional but not friendly.",
    "Salary is competitive for the market. Not high but not low either.",
    "The job security is decent. No major concerns but no guarantees either.",
    "Work is interesting enough. Some days are better than others.",
    "The company has its ups and downs. Overall a middle-of-the-road experience.",
    "Management listens sometimes. Other times they ignore employee concerns.",
    "The benefits package is adequate. Covers the basics but nothing extra.",
    "It's a good stepping stone. Not a career destination but a good start.",
    "The office is comfortable enough. Nothing fancy but it works.",
    "The team dynamics are okay. We get along but aren't close.",
    "Decent work environment. Some improvements needed but not critical.",
    "The company is stable. Not growing fast but not declining either.",
    "I can do my work without too many obstacles. That's good enough.",
    "The management style is middle-of-the-road. Not too strict, not too loose.",
    "Benefits could be better but they're not terrible. It's an average package.",
]

POSITIVE_TEMPLATES = [
    "Amazing company to work for! The culture is incredibly supportive and inclusive.",
    "Best job I've ever had. Great work-life balance and amazing benefits.",
    "The leadership team truly cares about employees. You feel valued here.",
    "Incredible growth opportunities. I've learned more here in one year than anywhere else.",
    "The company culture is outstanding. Everyone is friendly and collaborative.",
    "Great salary and benefits package. They really invest in their employees.",
    "The work is challenging and rewarding. Every day feels meaningful.",
    "Management is transparent and communicative. You always know where you stand.",
    "The team is fantastic. We work together seamlessly and support each other.",
    "Excellent work-life balance. Flexible hours and remote work options.",
    "The company invests heavily in employee development. Amazing training programs.",
    "I love working here. The environment is positive and energizing.",
    "The CEO is visionary and inspiring. Leadership sets a great example.",
    "Benefits are top-tier. Health, dental, vision, 401k match, and more.",
    "The office space is beautiful and well-designed. Very modern and comfortable.",
    "Promotion opportunities are real. Hard work is recognized and rewarded.",
    "The company values diversity and inclusion. It shows in everything they do.",
    "I feel like I'm making a real difference here. The work matters.",
    "The onboarding process was excellent. Felt supported from day one.",
    "Great communication from leadership. Town halls and regular updates keep everyone informed.",
    "The technology stack is cutting edge. Always working with the latest tools.",
    "Work-life balance is exceptional. Never expected to work overtime.",
    "The benefits keep getting better every year. Company is investing in people.",
    "I've been here 3 years and still love coming to work every day.",
    "The team is like family. We celebrate wins together and support through challenges.",
    "Management is approachable and supportive. Always available when you need them.",
    "The company has a clear vision and executes it well. Exciting time to be here.",
    "Excellent compensation packages. They pay above market rate for talent.",
    "The company culture is the best I've experienced. Truly people-first.",
    "I recommend this company to everyone. It's a genuinely great place to work.",
]

COMPANIES = [
    "TechCorp", "InnovateLab", "DataDrive", "CloudNine", "FutureSoft",
    "NexGen", "PulseAI", "QuantumLeap", "VantagePoint", "ApexDigital",
    "ZenithTech", "CipherLogic", "BrightPath", "VertexAI", "CatalystCo",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Portland, OR",
    "Miami, FL", "Atlanta, GA", "Remote", "London, UK",
]

JOB_TITLES = [
    "Software Engineer", "Data Scientist", "Product Manager", "UX Designer",
    "DevOps Engineer", "ML Engineer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Engineering Manager", "Data Analyst",
    "Technical Lead", "VP of Engineering", "CTO", "CEO",
]


def generate_review(sentiment: int) -> str:
    """Generate a realistic review for the given sentiment."""
    if sentiment == 0:
        base = random.choice(NEGATIVE_TEMPLATES)
    elif sentiment == 1:
        base = random.choice(NEUTRAL_TEMPLATES)
    else:
        base = random.choice(POSITIVE_TEMPLATES)

    # Add some variation
    prefixes = ["", "", "", "Honestly, ", "Overall, ", "In my experience, ", "To be honest, "]
    suffixes = ["", "", "", " Would not recommend.", " Glad I left.", " Still here though.", " It is what it is."]

    return random.choice(prefixes) + base + random.choice(suffixes)


def main():
    output_path = Path("data/employee_reviews.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Distribution: ~40% negative, ~20% neutral, ~40% positive (realistic for employee reviews)
    sentiments = [0] * 26600 + [1] * 13300 + [2] * 26600
    random.shuffle(sentiments)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "label"])

        for i, sentiment in enumerate(sentiments):
            review = generate_review(sentiment)
            writer.writerow([review, sentiment])

            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1}/{len(sentiments)} reviews")

    print(f"\nDataset saved to {output_path}")
    print(f"Total reviews: {len(sentiments)}")
    print(f"Distribution: Negative={sentiments.count(0)}, Neutral={sentiments.count(1)}, Positive={sentiments.count(2)}")


if __name__ == "__main__":
    main()
