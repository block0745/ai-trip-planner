import streamlit as st
from crewai import Agent, Task, Crew
from crewai.llms import OpenAI
from datetime import datetime, timedelta
import os

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="AI Trip Planner (OpenAI)",
    page_icon="✈️",
    layout="wide"
)

# =========================
# Initialize session state
# =========================
if "trip_plan" not in st.session_state:
    st.session_state.trip_plan = None
if "generating" not in st.session_state:
    st.session_state.generating = False

# =========================
# Create CrewAI Agents (OpenAI)
# =========================
def create_agents(openai_api_key: str):
    # OpenAI LLM (CrewAI)
    openai_llm = OpenAI(
        model="gpt-4o-mini",     # you can change to "gpt-4.1" if you want
        api_key=openai_api_key
    )

    # City Information Agent
    city_expert = Agent(
        role="City Information Expert",
        goal="Provide comprehensive information about cities including attractions, culture, and local tips",
        backstory=(
            "You are an experienced travel researcher with deep knowledge of cities worldwide. "
            "You excel at finding the most relevant and interesting information about destinations."
        ),
        verbose=True,
        allow_delegation=False,
        llm=openai_llm
    )

    # Itinerary Planner Agent
    itinerary_planner = Agent(
        role="Itinerary Planner",
        goal="Create detailed, personalized day-by-day travel itineraries based on user preferences",
        backstory=(
            "You are a professional travel planner with years of experience creating customized trips. "
            "You know how to balance activities, rest time, and local experiences perfectly."
        ),
        verbose=True,
        allow_delegation=False,
        llm=openai_llm
    )

    return city_expert, itinerary_planner

# =========================
# Create Tasks
# =========================
def create_tasks(
    city_expert,
    itinerary_planner,
    origin,
    destination,
    start_date,
    end_date,
    interests,
    budget,
    travel_style
):
    duration = (end_date - start_date).days + 1
    interests_str = ", ".join(interests)

    # Task 1: Research destination
    research_task = Task(
        description=f"""Research {destination} and provide comprehensive information including:
- Top 5 must-visit attractions related to: {interests_str}
- Best restaurants and local cuisine specialties
- Cultural highlights and local customs
- Transportation options within the city
- Weather considerations
- Important safety tips

Focus on {budget} budget options and {travel_style} travel style.
Keep the response concise and practical.""",
        agent=city_expert,
        expected_output="Detailed city information with practical travel tips in a structured format"
    )

    # Task 2: Create itinerary
    itinerary_task = Task(
        description=f"""Create a detailed {duration}-day itinerary for a trip from {origin} to {destination}.

Trip details:
- Dates: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}
- Interests: {interests_str}
- Budget: {budget}
- Travel style: {travel_style}

For each day provide:
- Morning (9 AM - 12 PM): Activities with specific locations
- Afternoon (12 PM - 6 PM): Activities with specific locations
- Evening (6 PM - 10 PM): Activities and dining suggestions
- Estimated daily cost range
- Transportation tips between locations

Format it clearly with day numbers and time blocks.
Make it practical, enjoyable, and realistic!""",
        agent=itinerary_planner,
        expected_output="Complete day-by-day itinerary with specific activities, timings, and practical details",
        context=[research_task]
    )

    return [research_task, itinerary_task]

# =========================
# Generate trip plan using CrewAI (OpenAI)
# =========================
def generate_trip_plan(
    openai_api_key,
    origin,
    destination,
    start_date,
    end_date,
    interests,
    budget,
    travel_style
):
    try:
        # Create agents
        city_expert, itinerary_planner = create_agents(openai_api_key)

        # Create tasks
        tasks = create_tasks(
            city_expert, itinerary_planner,
            origin, destination,
            start_date, end_date,
            interests, budget, travel_style
        )

        # Create crew
        crew = Crew(
            agents=[city_expert, itinerary_planner],
            tasks=tasks,
            verbose=True
        )

        # Execute
        result = crew.kickoff()
        return str(result)

    except Exception as e:
        return (
            f"Error generating trip plan: {str(e)}\n\n"
            "Please check your OpenAI API key and try again."
        )

# =========================
# UI Layout
# =========================
st.title("✈️ AI-Powered Trip Planner (OpenAI)")
st.markdown("### Plan Your Perfect Trip with CrewAI & OpenAI")

# =========================
# Sidebar for API key
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("**Get your OpenAI API key:**")
    st.markdown("1. Go to the OpenAI platform")
    st.markdown("2. Create an API key")
    st.markdown("3. Paste it below")

    # Optional: read from environment variable if present
    env_key = os.getenv("OPENAI_API_KEY", "")

    openai_api_key = env_key or st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )

    if openai_api_key:
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Please enter your OpenAI API key")

    st.markdown("---")
    st.markdown("### About")
    st.info("This app uses CrewAI agents powered by OpenAI to create personalized trip plans.")

# =========================
# Main content
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Trip Details")

    origin = st.text_input("Origin City", placeholder="e.g., New York", value="New York")
    destination = st.text_input("Destination City", placeholder="e.g., Paris", value="Paris")

    st.write("**Travel Dates**")
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() + timedelta(days=7),
        min_value=datetime.now()
    )

    end_date = st.date_input(
        "End Date",
        value=datetime.now() + timedelta(days=12),
        min_value=start_date
    )

    duration = (end_date - start_date).days + 1
    st.info(f"📅 Trip Duration: {duration} days")

with col2:
    st.subheader("🎯 Preferences")

    interests = st.multiselect(
        "Your Interests",
        [
            "Culture & Museums", "Food & Dining", "Adventure & Sports",
            "Nature & Parks", "Shopping", "Nightlife", "History",
            "Beach & Water Activities", "Art & Architecture"
        ],
        default=["Culture & Museums", "Food & Dining"]
    )

    budget = st.select_slider(
        "Budget Range",
        options=["Budget", "Moderate", "Comfortable", "Luxury"],
        value="Moderate"
    )

    travel_style = st.radio(
        "Travel Style",
        ["Relaxed", "Balanced", "Packed"],
        index=1,
        horizontal=True
    )

# =========================
# Generate button
# =========================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    generate_button = st.button("🚀 Generate AI Trip Plan", type="primary", use_container_width=True)

    if generate_button:
        if not openai_api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        elif not origin or not destination:
            st.error("⚠️ Please enter both origin and destination cities.")
        elif not interests:
            st.error("⚠️ Please select at least one interest.")
        else:
            st.session_state.generating = True

            with st.status("🤖 AI Agents are planning your trip...", expanded=True) as status:
                st.write("🔍 City Expert is researching your destination...")
                st.write("📝 Itinerary Planner is creating your schedule...")
                st.write("⚡ Using OpenAI models...")

                result = generate_trip_plan(
                    openai_api_key, origin, destination, start_date, end_date,
                    interests, budget, travel_style
                )

                st.session_state.trip_plan = result
                st.session_state.generating = False

                status.update(
                    label="✅ Trip plan generated successfully!",
                    state="complete",
                    expanded=False
                )

# =========================
# Display results
# =========================
if st.session_state.trip_plan:
    st.success("✅ Your personalized trip plan is ready!")

    st.markdown("### 📋 Trip Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Route", f"{origin} → {destination}")
        st.metric("Duration", f"{duration} days")

    with summary_col2:
        st.metric("Budget", budget)
        st.metric("Style", travel_style)

    st.info(f"**Interests:** {', '.join(interests)}")
    st.caption(f"**Travel Dates:** {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}")

    st.markdown("### 🗺️ Your Personalized Trip Plan")

    with st.container():
        st.markdown(st.session_state.trip_plan)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.download_button(
            label="📥 Download Plan",
            data=st.session_state.trip_plan,
            file_name=f"trip_plan_{destination.lower().replace(' ', '_')}_{start_date.strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        if st.button("🔄 Generate New Plan", use_container_width=True):
            st.session_state.trip_plan = None
            st.rerun()

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <p>⚡ Built with Streamlit & CrewAI | Powered by OpenAI</p>
</div>
""",
    unsafe_allow_html=True
)
