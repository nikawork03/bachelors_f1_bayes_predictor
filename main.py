import requests

URL = 'https://worldcupjson.net/'


# A delimeter function that helps us to make the program more intuitive and easy to read for users.
def delimeter(num):
    return f"{num * '-'}"


# If live games are at the moment then returns information about it.
def live_games(url_path):
    try:
        r = requests.get(URL + url_path)
        content = r.json()

        if len(content) < 1:
            print(f"\nResult: There are no live games at the moment")
            print(delimeter(50))
        else:
            print(delimeter(50))
            for game in content:
                home_team = game['home_team']
                away_team = game['away_team']
                time = game['time']
                print(
                    f"Time: {time} {home_team['name']} {home_team['goals']} : {away_team['goals']} {away_team['name']}")
            print(delimeter(50))

    except requests.exceptions.HTTPError as errh:
        print(f'HTTP error')
        print(errh.args[0])

    except requests.exceptions.MissingSchema as errmiss:
        print("Missing schema: include http or https", errmiss)

    except:
        print('No Information Found')


# If games are scheduled for today, it will return information about those games.
def today_games(url_path):
    try:
        r = requests.get(URL + url_path)
        content = r.json()

        if len(content) < 1:
            print(f"\nResult: There are no games today")
            print(delimeter(50))
        else:
            print(delimeter(50))
            for index, game in content:
                home_team = game['home_team']
                away_team = game['away_team']

                if home_team['goals'] != None:

                    home_goal = home_team['goals']
                else:
                    home_goal = '-'

                if away_team['goals'] != None:
                    away_goal = away_team['goals']
                else:
                    away_goal = '-'
                print()
                print(f"{index}. {home_team['name']} {home_goal} : {away_goal} {away_team['name']}")
            print(delimeter(50))
    except:
        print("No Information Found")

    # Prints all the information about all matches.


def all_games(url_path):
    try:
        r = requests.get(URL + url_path)
        content = r.json()

        print(delimeter(50))
        index = 1
        for game in content:
            home_team = game['home_team']
            away_team = game['away_team']

            if home_team['goals'] != None:
                home_goal = home_team['goals']
            else:
                home_goal = '-'

            if away_team['goals'] != None:
                away_goal = away_team['goals']
            else:
                away_goal = '-'

            print(
                f"{index}.  {home_team['name'].ljust(15, ' ')} {home_goal}  :  {away_goal} {away_team['name'].rjust(18, ' ')}")
            print(delimeter(50))
            index += 1
    except:
        print(delimeter(50))
        print('No Information Found')

    # Prints the table which consists each team's points, wins, draws, losses, goals, goals_against, goal_differential


def group_standings(url_path):
    try:
        r = requests.get(URL + url_path)
        content = r.json()

        groups = content['groups']

        for group in groups:
            print(delimeter(50))
            print(f"Group: {group['letter'].ljust(16, ' ')} P.  W.  D.  L.  G.  GA. DF.")

            teams = group['teams']
            index = 1
            for country in teams:
                print(
                    f"{index}. {country['name'].ljust(20, ' ')} {country['group_points']} | {country['wins']} | {country['draws']} | {country['losses']} | {country['goals_for']} | {country['goals_against']} | {country['goal_differential']}")
                index += 1
        print(delimeter(50))
    except:
        print('No Information Found')

    # Returns all 64 games general information, infromation includes(Date, Location, Stadium, Attendance)


def games_general_information(url_path):
    try:
        r = requests.get(URL + url_path)
        content = r.json()

        for game in content:
            location = game['location']
            attendance = game['attendance']
            date = game['datetime'][:10]
            stadium = game['venue']
            home_team = game['home_team']
            away_team = game['away_team']
            print(delimeter(125))
            print(
                f"{home_team['name']} VS {away_team['name']} | Date: {date} | Location: {location} | Stadium: {stadium} | Attendance: {attendance}")
        print(delimeter(125))
    except:
        print("No Information Found.")

    # Menu function where we allow the user to select the information he wants and according to the selected information, we run the corresponding function


def menu():
    print('World Cup Qatar 2022:')
    while True:
        print('1. Live matches')
        print("2. Today's matches")
        print('3. All games')
        print('4. Standings')
        print('5. Games general information')

        user_choice = input("Choose option(1-5) or 'q' for exit: ").lower()

        if user_choice == "q":
            break
        elif user_choice == '1':
            live_games('matches/current')
        elif user_choice == '2':
            today_games('matches/today')
        elif user_choice == '3':
            all_games('matches')
        elif user_choice == '4':
            group_standings('teams')
        elif user_choice == '5':
            games_general_information('matches/?by=total_goals')
        else:
            print(delimeter(50))
            print(f'{user_choice} is not valid input, please try again: ')
            print(delimeter(50))


if __name__ == "__main__":
    menu()
