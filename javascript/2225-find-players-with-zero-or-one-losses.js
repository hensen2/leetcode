/**
 * 2225 - Find Players With Zero or One Losses
 * Hashmap | Count frequencies
 * Time: O(n*log(n)) | Space: O(n)
 * Time complexity is O(n*log(n)) for sorting
 * @param {number[][]} matches
 * @return {number[][]}
 */
const findWinners = function (matches) {
  const lossCounts = new Map();

  for (const [winner, loser] of matches) {
    lossCounts.set(winner, lossCounts.get(winner) || 0);
    lossCounts.set(loser, (lossCounts.get(loser) || 0) + 1);
  }

  const zeroLoss = [];
  const oneLoss = [];

  for (const [player, count] of lossCounts) {
    if (count === 0) zeroLoss.push(player);
    if (count === 1) oneLoss.push(player);
  }

  return [zeroLoss.sort((a, b) => a - b), oneLoss.sort((a, b) => a - b)];
};
